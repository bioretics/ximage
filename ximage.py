#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import os, sys, argparse, cv2, ast
from libxmp import XMPFiles, XMPMeta, XMPError, XMPIterator, consts
from uuid import uuid4, UUID
from hashlib import sha1
from datetime import datetime
from string import Template
import pickle

XMP_NS_ALIQUIS = 'http://bioretics.com/aliquis'

XMPMeta.register_namespace(XMP_NS_ALIQUIS, 'aliquis')

__all__ = [ 'XImageMeta', 'XItem', 'XClass', 'XBlob', 'XImageParseError', 'XImageEmptyXMPError', 'ximread', 'ximwrite', 'ximage_main' ]

if sys.version_info[0] == 3:
    def raise_(exc, tb=None):
        if exc.__traceback__ is not tb:
            raise exc.with_traceback(tb)
        raise exc
else:
    exec('def raise_(exc, tb=None):\n    raise exc, None, tb\n')

class XImageEmptyXMPError(Exception):
    def __init__(self, file_path):
        self.file_path = file_path

    def __str__(self):
        return 'empty XMP in file "%s"' % (self.file_path,)

class XImageParseError(Exception):
    def __init__(self, tag_name):
        self.tag_name = tag_name

    def __str__(self):
        return 'parsing tag "%s"' % (self.tag_name,)

class XImageMeta(object):
    XMP_TEMPLATE = """<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Exempi + XMP Core 5.1.2">
     <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
      <rdf:Description rdf:about="" xmlns:aliquis="http://bioretics.com/aliquis">
       <aliquis:acquisition>%(acquisition)s</aliquis:acquisition>
       <aliquis:setup>%(setup)s</aliquis:setup>
       <aliquis:classes>%(classes)s</aliquis:classes>
       <aliquis:items>%(items)s</aliquis:items>
      </rdf:Description>
     </rdf:RDF>
    </x:xmpmeta>"""

    def __init__(self, classes, items=None, acquisition=None, setup=None):
        self.classes = classes
        self.items = [] if items is None else items
        self.acquisition = {} if acquisition is None else acquisition
        self.setup = {} if setup is None else setup

    def get_colormap(self):
        return [ c.color for c in self.classes ]

    def to_xmp(self):
        xmp = XMPMeta()
        xmp.parse_from_str(str(self))
        return xmp

    def write(self, path):
        xmpfile = XMPFiles(file_path=path, open_forupdate=True)
        xmp = self.to_xmp()
        #assert xmpfile.can_put_xmp(xmp)
        xmpfile.put_xmp(xmp)
        xmpfile.close_file()

    @staticmethod
    def read(path):
        xmpfile = XMPFiles(file_path=path, open_forupdate=False)
        xmp = xmpfile.get_xmp()
        if xmp is None:
            raise XImageEmptyXMPError(path)
        return XImageMeta.parse(xmp)

    @staticmethod
    def parse(xmp_or_str):
        if type(xmp_or_str) == str:
            xmp = XMPMeta()
            xmp.parse_from_str(xmp_or_str)
        else:
            xmp = xmp_or_str

        try:
            attribs = set([ x[1][8:] for x in XMPIterator(xmp, XMP_NS_ALIQUIS) if x[1].startswith('aliquis:') ])
            tag = 'acquisition'
            acquisition = XImageMeta.parse_dict(xmp, tag) if '%s[1]' % (tag,) in attribs else {}
            tag = 'setup'
            setup = XImageMeta.parse_dict(xmp, tag) if '%s[1]' % (tag,) in attribs else {}
            tag = 'classes'
            classes = [ XClass.parse(xmp, '%s[%d]' % (tag, i)) for i in range(1, 1 + xmp.count_array_items(XMP_NS_ALIQUIS, tag)) ] if '%s[1]' % (tag,) in attribs else []
            tag = 'items'
            items = [ XItem.parse(xmp, '%s[%d]' % (tag, i)) for i in range(1, 1 + xmp.count_array_items(XMP_NS_ALIQUIS, tag)) ] if '%s[1]' % (tag,) in attribs else []
        except:
            raise_(XImageParseError(tag), sys.exc_info()[2])

        return XImageMeta(classes, items, acquisition, setup)

    @staticmethod
    def parse_value(xmp, prefix):
        t = (xmp.get_property(XMP_NS_ALIQUIS, '%s/aliquis:type' % prefix)).lower()
        if t.startswith('datetime'):
            return xmp.get_property_datetime(XMP_NS_ALIQUIS, '%s/aliquis:value' % prefix)

        y = xmp.get_property(XMP_NS_ALIQUIS, '%s/aliquis:value' % prefix)
        if t.startswith('bool'):
            y = bool(int(y))
        elif t.startswith('int'):
            y = int(y)
        elif t.startswith('float'):
            y = float(y)

        return y

    @staticmethod
    def str_value(v):
        if type(v) == bool:
            t = 'boolean'
            v = 1 if v else 0
        elif type(v) == int:
            t = 'integer'
        elif type(v) == float:
            t = 'float'
        elif type(v) == datetime:
            t = 'datetime'
            v = v.strftime('%Y-%m-%dT%H:%M:%S')
        else:
            t = 'string'
        return '<aliquis:type>%s</aliquis:type><aliquis:value>%s</aliquis:value>' % (t, str(v))

    @staticmethod
    def parse_list(xmp, prefix):
        if xmp.does_property_exist(XMP_NS_ALIQUIS, prefix):
            return [ XImageMeta.parse_value(xmp, '%s[%d]' % (prefix, i)) for i in range(1, 1 + xmp.count_array_items(XMP_NS_ALIQUIS, prefix)) ]
        return []

    @staticmethod
    def str_list(l):
        if len(l) == 0:
            return ''
        return '<rdf:Seq>%s</rdf:Seq>' % (''.join([ '<rdf:li rdf:parseType="Resource">%s</rdf:li>' % XImageMeta.str_value(x) for x in l ]))

    @staticmethod
    def parse_dict(xmp, prefix):
        if xmp.does_property_exist(XMP_NS_ALIQUIS, prefix):
            return { xmp.get_property(XMP_NS_ALIQUIS, '%s[%d]/aliquis:name' % (prefix, i)): XImageMeta.parse_value(xmp, '%s[%d]' % (prefix, i)) for i in range(1, 1 + xmp.count_array_items(XMP_NS_ALIQUIS, prefix)) }
        return {}

    @staticmethod
    def str_dict(d):
        if len(d) == 0:
            return ''
        return '<rdf:Bag>%s</rdf:Bag>' % (''.join([ '<rdf:li rdf:parseType="Resource"><aliquis:name>%s</aliquis:name>%s</rdf:li>' % (k, XImageMeta.str_value(v)) for k, v in d.items() ]))

    def __str__(self):
        acquisition_str = XImageMeta.str_dict(self.acquisition)
        setup_str = XImageMeta.str_dict(self.setup)
        classes_str = '<rdf:Seq>%s</rdf:Seq>' % (''.join([ '<rdf:li rdf:parseType="Resource">%s</rdf:li>' % str(c) for c in self.classes ]))
        items_str = '<rdf:Bag>%s</rdf:Bag>' % (''.join([ '<rdf:li rdf:parseType="Resource">%s</rdf:li>' % str(item) for item in self.items ])) if len(self.items) > 0 else ''
        return XImageMeta.XMP_TEMPLATE % { 'acquisition': acquisition_str, 'setup': setup_str, 'classes': classes_str, 'items': items_str }

class XItem(object):
    XMP_TEMPLATE = '<aliquis:uuid>%(uuid)s</aliquis:uuid><aliquis:blobs><rdf:Bag>%(blobs)s</rdf:Bag></aliquis:blobs>'

    def __init__(self, blobs, uuid=None):
        assert len(blobs) > 0, 'An item must contain at least one blob'
        self.blobs = blobs
        self.uuid = uuid4() if uuid is None else uuid

    @staticmethod
    def parse(xmp, prefix):
        uuid = UUID(xmp.get_property(XMP_NS_ALIQUIS, '%s/aliquis:uuid' % prefix))
        blobs = [ XBlob.parse(xmp, '%s/aliquis:blobs[%d]' % (prefix, i)) for i in range(1, 1 + xmp.count_array_items(XMP_NS_ALIQUIS, '%s/aliquis:blobs' % prefix)) ]
        return XItem(blobs, uuid)

    def __str__(self):
        return XItem.XMP_TEMPLATE % { 'uuid': str(self.uuid), 'blobs': ''.join([ '<rdf:li rdf:parseType="Resource">%s</rdf:li>' % str(blob) for blob in self.blobs ]) }

class XClass(object):
    XMP_TEMPLATE = '<aliquis:name>%(name)s</aliquis:name><aliquis:color>%(color)s</aliquis:color>'

    def __init__(self, name, color=None, remap=None):
        self.name = name
        self.color = color or XClass.get_random_color()
        self.remap = remap

    @staticmethod
    def get_random_color():
        return tuple(np.random.randint(0, 256, 3).tolist())

    @staticmethod
    def parse(xmp, prefix):
        name = xmp.get_property(XMP_NS_ALIQUIS, '%s/aliquis:name' % prefix)
        #try:
        color = tuple(map(int, xmp.get_property(XMP_NS_ALIQUIS, '%s/aliquis:color' % prefix).split(',')))
        #except:
        #    color = None

        try:
            remap = int(xmp.get_property(XMP_NS_ALIQUIS, '%s/aliquis:remap' % prefix))
        except:
            remap = None

        return XClass(name, color, remap)

    def __eq__(self, other):
        return self.name == other.name and self.color == other.color and self.remap == other.remap

    def __str__(self):
        s = XClass.XMP_TEMPLATE % { 'name': str(self.name), 'color': ','.join(map(str, self.color)) }
        if self.remap is not None:
            s += '<aliquis:remap>%d</aliquis:remap>' % self.remap
        return s

class XBlob(object):
    XMP_TEMPLATE = '<aliquis:values>%(values)s</aliquis:values><aliquis:points>%(points)s</aliquis:points>%(blobs)s'

    def __init__(self, points, values, children=None):
        self.points = points
        self.values = values
        self.children = [] if children is None else children

    def get_classid(self):
        return np.argmax(self.values)

    def get_contour_area(self):
        return float(cv2.contourArea(self.points))

    def get_area(self):
        # Using masks is (far) more accurate
        return self.get_contour_area() - sum([ b.get_contour_area() for b in self.children ])

    def draw(self, im, colormap, filled=False):
        classid = self.get_classid()
        color_alpha = colormap[classid]
        color = tuple(color_alpha[:3])
        if filled:
            if len(color_alpha) == 4:
                overlay = im.copy()
                alpha = color_alpha[3] / 255.0
                cv2.fillPoly(overlay, [ self.points ], color)
                cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0, im)
            else:
                cv2.fillPoly(im, [ self.points ], color)
        else:
            cv2.drawContours(im, [ self.points ], 0, color)

        for blob in self.children:
            blob.draw(im, colormap, filled)

        return im

    def get_mask(self, shape, dtype=np.uint8):
        mask = np.zeros(shape, dtype=dtype)
        return self.draw(mask, { self.get_classid(): (1,) }, True)

    def get_mask_like(self, im):
        return self.get_mask(im.shape, im.dtype)

    @staticmethod
    def parse(xmp, prefix):
        points = np.int32(list(map(int, xmp.get_property(XMP_NS_ALIQUIS, '%s/aliquis:points' % prefix).split(','))))
        values = np.float32(list(map(float, xmp.get_property(XMP_NS_ALIQUIS, '%s/aliquis:values' % prefix).split(','))))

        if xmp.does_property_exist(XMP_NS_ALIQUIS, '%s/aliquis:blobs' % prefix):
            children = [ XBlob.parse(xmp, '%s/aliquis:blobs[%d]' % (prefix, i)) for i in range(1, 1 + xmp.count_array_items(XMP_NS_ALIQUIS, '%s/aliquis:blobs' % prefix)) ]
        else:
            children = []

        return XBlob(points.reshape(len(points) // 2, 2), values, children)

    def __str__(self):
        values_str = ','.join(map(str, self.values))
        points_str = ','.join(map(str, self.points.flatten()))
        children_str = '<aliquis:blobs><rdf:Bag>%s</rdf:Bag></aliquis:blobs>' % (''.join([ '<rdf:li rdf:parseType="Resource">%s</rdf:li>' % str(child) for child in self.children ]))
        return XBlob.XMP_TEMPLATE % { 'values': values_str, 'points': points_str, 'blobs': children_str if len(self.children) > 0 else '' }

def ximread(path):
    im = cv2.imread(path, -1)
    assert im is not None, 'Image data missing'
    meta = XImageMeta.read(path)
    return im, meta

def ximwrite(path, im, meta):
    cv2.imwrite(path, im)
    meta.write(path)

################################################################################
# XImage utility functions #####################################################
################################################################################

def ximage_inject(args):
    with open(args.metadata, 'r') as f:
        m = XImageMeta.parse(f.read())
    m.write(args.path)
    return 0

def ximage_extract(args):
    print(str(XImageMeta.read(args.path)))
    return 0

def ximage_uuid(args):
    meta = XImageMeta.read(args.path)
    sorted_items = sorted(meta.items, key=lambda item: np.vstack([ b.points for b in item.blobs ]).mean(axis=0).round().astype(int).tolist())
    if len(args.uuids) == 0:
        for item in sorted_items:
            print(str(item.uuid))
    else:
        assert len(args.uuids) == len(sorted_items), 'UUIDs must be %d' % (len(sorted_items),)
        for uuid, item in zip(args.uuids, sorted_items):
            if uuid == '0':
                continue
            item.uuid = UUID(uuid)
        meta.write(args.path)
    return 0

def ximage_import(args):
    def versor(d, shape, dtype=np.float32):
        v = np.zeros(shape, dtype=dtype)
        v[d] = 1
        return v

    def contour_level(hier, i, l=0):
        _, _, child, parent = hier[i]
        if parent == -1:
            return l
        return contour_level(hier, parent, l + 1)

    def blob_init_data(blob, template_mask):
        mask = blob.get_mask_like(template_mask)
        return (mask, float(np.count_nonzero(mask)))

    def all_subblobs(blob, blobs, blobs_data, overlap_ratio_threshold):
        subblobs = []
        blob_mask, blob_area = blobs_data[blob]
        for b in blobs - set([ blob ]):
            #if 1 - np.count_nonzero(blobs_data[b][0] * blob_mask != blobs_data[b][0]) / blobs_data[b][1] >= overlap_ratio_threshold:
            if np.all(blobs_data[b][0] * blob_mask == blobs_data[b][0]):
                subblobs.append(b)
        return set(subblobs)

    def blob_descendents(blobs_subblobs, blob):
        descendents = subblobs = blobs_subblobs[blob]
        for b in subblobs:
            descendents = descendents.union(blob_descendents(blobs_subblobs, b))
        return descendents

    def build_hierarchy(blobs_children, parent):
        # Find and remove roots from graph edges
        roots = set(blobs_children.keys()) - set([ x for xs in blobs_children.values() for x in xs ])
        for root in roots:
            blobs_children.pop(root)

        # Recursive step
        for root in roots:
            parent.children.append(root)
            build_hierarchy(blobs_children, root)

    mask = cv2.imread(args.mask, -1)

    # Count number of classes
    classes_count = len(np.trim_zeros(np.bincount(mask.flatten())[:-1], 'b')) - 1
    if len(args.classes) > 0:
        classes = list(map(XClass, args.classes))
    else:
        classes = [ XClass(str(i)) for i in range(classes_count) ]
    classes_num = len(classes)
    assert classes_num >= classes_count, 'Classes must be at least %d' % (classes_count)

    for xc, color in zip(classes, args.colors):
        try:
            xc.color = _COLORS[color]
        except KeyError:
            color = color.strip()[2:]
            xc.color = tuple([ int(color[i:(i + 2)], 16) for i in range(0, 6, 2) ])

    #
    default_class = 0
    overlap_ratio_threshold = 0.9
    default_value = versor(default_class, classes_num)

    #
    ret = cv2.findContours(np.pad(mask != 255, 1, 'constant', constant_values=0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = ret[-2]

    #
    items = [ XItem([ XBlob(contour.squeeze(1) - 1, default_value) ]) for contour in contours ]

    # Find items subblobs
    for item in items:
        item_blob = item.blobs[0]
        item_mask = item_blob.get_mask_like(mask)

        # Find all subblobs inside the item
        blobs = set()
        item_indexmask = item_mask * mask
        for c in range(classes_num):
            if c == default_class:
                continue

            ret = cv2.findContours((item_indexmask == c).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contours = ret[-2]
            hier = ret[-1]

            # Blobs are only even levels (contours hierarchy alternate full and empty areas)
            blobs = blobs.union(set([ XBlob(contour.squeeze(1), versor(c, classes_num)) for i, contour in enumerate(contours) if contour_level(hier[0], i) % 2 == 0 ]))

        # b: (mask, area)
        blobs_data = { b: blob_init_data(b, mask) for b in blobs }

        # b: [ blobs contained in b ]
        blobs_subblobs = { b: all_subblobs(b, blobs, blobs_data, overlap_ratio_threshold) for b in blobs }

        # b: [ b's children ]
        blobs_children = { b: subblobs - reduce(set.union, [ blob_descendents(blobs_subblobs, x) for x in subblobs ], set()) for b, subblobs in blobs_subblobs.items() }

        # Reconstruct blobs hierarchy for the item
        item_blob.children = list(blobs - reduce(set.union, blobs_children.values(), set()))
        for blob in blobs:
            blob.children = list(blobs_children[blob])

    for item, uuid in zip(items, args.uuids):
        if uuid == '0':
            continue
        item.uuid = UUID(uuid)

    im_meta = XImageMeta(classes, items)
    im_meta.write(args.path)
    return 0

def ximage_export(args):
    im, im_meta = ximread(args.path)
    colormap = { i: (i if c.remap is None else c.remap,) for i, c in enumerate(im_meta.classes) }
    mask = np.full(im.shape[:2], 255, dtype=np.uint8)
    for item in im_meta.items:
        for blob in item.blobs:
            blob.draw(mask, colormap, True)
    cv2.imwrite(args.mask, mask)
    return 0

def ximage_update(args):
    im_path = args.path
    overwrite = args.overwrite
    im_meta = XImageMeta.read(im_path)

    mapping = dict([ kv.split('=') for kvs in args.mapping for kv in kvs.split() ])
    with open(args.metadata, 'r') as f:
        im_meta_update = XImageMeta.parse(Template(f.read()).substitute(mapping))

    if args.replace_classes:
        im_meta.classes = im_meta_update.classes
    else:
        classes = im_meta.classes
        classes_num = len(classes)
        classes_update = im_meta_update.classes
        if classes_num == 0 or (len(classes_update) >= classes_num and all([ c.name == cu.name for c, cu in zip(classes, classes_update) ])):
            if overwrite:
                for c, cu in zip(classes, classes_update):
                    c.color = cu.color
            classes.extend(classes_update[classes_num:])

    acquisition = im_meta.acquisition
    for a_name, a in im_meta_update.acquisition.items():
        if overwrite or (a_name not in acquisition):
            acquisition[a_name] = a

    setup = im_meta.setup
    for s_name, s in im_meta_update.setup.items():
        if overwrite or (s_name not in setup):
            setup[s_name] = s

    im_meta.write(im_path)

def ximage_view(args):
    im_path = args.path

    if args.metadata:
        with open(args.metadata, 'r') as f:
            im_meta = XImageMeta.parse(f.read())
    else:
        im_meta = XImageMeta.read(im_path)

    items = im_meta.items
    colormap = im_meta.get_colormap()

    # Display infos on terminal
    sys.stderr.write('Acquisition parameters:\n')
    for k, v in sorted(im_meta.acquisition.items()):
        sys.stderr.write('- %s: %s\n' % (k, str(v)))

    sys.stderr.write('Setup parameters:\n')
    for k, v in sorted(im_meta.setup.items()):
        sys.stderr.write('- %s: %s\n' % (k, str(v)))

    sys.stderr.write('Image contain %d item%s.\n' % (len(items), 's' if len(items) != 1 else ''))

    # Debug draw (if image available)
    im = cv2.imread(im_path, -1) if im_path is not None else None
    if im is not None:
        # Create debug image as a color copy of im
        im_debug = np.zeros(im.shape[:2] + (3,), dtype=np.uint8)
        if im.ndim == 2:
            im_debug[:, :, 0] = im_debug[:, :, 1] = im_debug[:, :, 2] = im
        else:
            im_debug[:, :, :3] = im[:, :, :3]

        # Draw items blobs
        for item in items:
            for blob in item.blobs:
                blob.draw(im_debug, colormap)
                uuid_text = str(item.uuid)
                (uuid_w, uuid_h), uuid_baseline = cv2.getTextSize(uuid_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                blob_topleft = blob.points.min(axis=0).round().astype(int)
                blob_y = blob_topleft[1] - uuid_h + uuid_baseline - 4
                if blob_y < 4:
                    blob_bottomright = blob.points.max(axis=0).round().astype(int)
                    blob_y = blob_bottomright[1] + uuid_h + uuid_baseline + 4
                blob_center = blob.points.mean(axis=0).round().astype(int)
                cv2.putText(im_debug, uuid_text, (blob_center[0] - uuid_w // 2, blob_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0xff, 0xff, 0xff))

        # Write infos on debug image
        x = 15
        for class_id, xclass in enumerate(im_meta.classes):
            class_color = xclass.color
            y = (class_id + 1) * 25
            cv2.line(im_debug, (x, y - 5), (x + 20, y - 5), class_color)
            cv2.putText(im_debug, xclass.name, (x + 30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0xff, 0xff, 0xff))
    else:
        im_debug = None

    if im_debug is not None:
        cv2.imshow(im_path, im_debug)
        cv2.waitKey(0)

    return 0

class XValue(object):
    def __init__(self, val):
        self.val = val

    @staticmethod
    def parse(buf):
        return pickle.loads(str(buf))

    def __str__(self):
        return str(self.val)

def _ximage_index_connect(args, create=False):
    if 'sqlite3' not in sys.modules:
        import sqlite3

        # Custom database types converters and adapters
        sqlite3.register_converter('xvalue', XValue.parse)
        sqlite3.register_converter('color', lambda buf: tuple(np.frombuffer(buf, dtype='|u1').tolist()))
        sqlite3.register_converter('vector', lambda buf: np.frombuffer(buf, dtype='<f4'))
        sqlite3.register_converter('points', lambda buf: np.frombuffer(buf, dtype='<i4').reshape(len(buf) / 8, 2))
        sqlite3.register_converter('uuid', lambda buf: UUID(bytes=buf))
        sqlite3.register_adapter(XValue, lambda x: pickle.dumps(x.val))
        sqlite3.register_adapter(np.ndarray, lambda a: np.getbuffer(a))
        sqlite3.register_adapter(UUID, lambda uuid: buffer(uuid.get_bytes()))

    index_path = os.path.join(args.root, '.ximage-index.db')
    if not create:
        # Try to open existing database path (raise IOError)
        with open(index_path, 'r') as f:
            pass

    conn = sqlite3.connect(index_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.create_function('xvalue_parse', 1, XValue.parse)
    return conn

def _ximage_index_insert_blobs(cur, blob, classes, xbelonging_id, parent_id=None):
    cid = blob.get_classid()
    c = classes[cid]
    (xclass_id,) = cur.execute('SELECT id FROM XClass WHERE classid=? AND name=? AND color=?', (cid, c.name, np.array(c.color, dtype=np.uint8))).fetchone()
    cur.execute('INSERT OR REPLACE INTO XBlob(xbelonging_id, parent_id, xclass_id, val, area, vals, contour) VALUES(?, ?, ?, ?, ?, ?, ?)', (xbelonging_id, parent_id, xclass_id, float(blob.values[cid]), blob.get_area(), blob.values, blob.points))
    blob_id = cur.lastrowid
    for b in blob.children:
        _ximage_index_insert_blobs(cur, b, classes, xbelonging_id, blob_id)

def ximage_index(args):
    TABLES_SEARCH = [ 'XBlob', 'XItem', 'XBelonging', 'XImage', 'XClass', 'XImageParam' ]
    IMAGES_EXTS = [ '.png', '.tif', '.tiff', '.jpg', '.jpeg' ]

    root = os.path.realpath(args.root)

    try:
        conn = _ximage_index_connect(args, True)
    except ImportError:
        sys.stderr.write('Error: cannot import sqlite3 module\n')
        return -1

    cur = conn.cursor()
    cur.execute('SELECT * FROM sqlite_master WHERE type="table" AND name IN (%s);' % (','.join(map(repr, TABLES_SEARCH)),))
    if len(cur.fetchall()) != len(TABLES_SEARCH):
        cur.executescript(_XIMAGE_INDEX_CREATE_SCHEMA)
        conn.commit()

    ims_ids = {}
    for root_path, _, filenames in os.walk(root):
        for im_filename in filter(lambda f: os.path.splitext(f)[1].lower() in IMAGES_EXTS, filenames):
            im_path = os.path.realpath(os.path.join(root_path, im_filename))
            im_relpath = os.path.relpath(im_path, root)
            try:
                im, im_meta = ximread(im_path)
                im.flags.writeable = False
                im_id = UUID(bytes=sha1(im.data).digest()[:16], version=4)
                try:
                    sys.stderr.write('Error: inserting %s: duplicate image (%s)\n' % (im_relpath, ims_ids[im_id]))
                    continue
                except KeyError:
                    pass

                # Update XClasses
                for i, c in enumerate(im_meta.classes):
                    cur.execute('INSERT OR IGNORE INTO XClass(classid, name, color) VALUES(?, ?, ?)', (i, c.name, np.array(c.color, dtype=np.uint8)))
                conn.commit()

                # Update Acquisition

                for name, val in im_meta.acquisition.items():
                    cur.execute('INSERT OR IGNORE INTO XImageParam(ximage_id, param_type, name, val) VALUES(?, 0, ?, ?)', (im_id, name, XValue(val)))
                conn.commit()

                # Update Setup
                for name, val in im_meta.setup.items():
                    cur.execute('INSERT OR IGNORE INTO XImageParam(ximage_id, param_type, name, val) VALUES(?, 1, ?, ?)', (im_id, name, XValue(val)))
                conn.commit()

                # Insert XImage
                cur.execute('INSERT OR REPLACE INTO XImage(id, path) VALUES(?, ?)', (im_id, im_relpath))

                # Insert XItems
                for item in im_meta.items:
                    cur.execute('INSERT OR REPLACE INTO XItem(id) VALUES(?)', (item.uuid,))
                    cur.execute('INSERT OR REPLACE INTO XBelonging(ximage_id, xitem_id) VALUES(?, ?)', (im_id, item.uuid))
                    xbelonging_id = cur.lastrowid
                    for blob in item.blobs:
                        _ximage_index_insert_blobs(cur, blob, im_meta.classes, xbelonging_id)

                # Commit insert
                conn.commit()
                sys.stderr.write('Done %s\n' % (im_relpath,))
                ims_ids[im_id] = im_relpath
            except Exception as e:
                sys.stderr.write('Error: inserting %s: %s\n' % (im_relpath, str(e)))

    return 0

def ximage_query(args):
    class XEvalContext(object):
        def __init__(self, cur):
            self.cur = cur
            self.cur.execute('SELECT path FROM XImage;')
            self.all_paths = self._fetch_all()
            self.reset()

        def push_param(self, p):
            n = 'x%d' % (len(self.params),)
            self.params[n] = p
            return ':%s' % (n,)

        def execute_query(self):
            from_clause = ', '.join(self.from_tables)
            where_clause = ' AND '.join(self.where_conjs)
            groupby_clause = '' if len(self.having_conjs) == 0 else ' GROUP BY path HAVING %s' % (' AND '.join(self.having_conjs),)
            query = 'SELECT path FROM %s WHERE %s%s;' % (from_clause, where_clause, groupby_clause)
            #print query, self.params
            self.cur.execute(query, self.params)
            return self._fetch_all()

        def reset(self):
            self.params = {}
            self.where_conjs = set()
            self.from_tables = set()
            self.having_conjs = set()

        def _fetch_all(self):
            return set([ r[0] for r in self.cur.fetchall() ])

    def xeval_num(node):
        return node.n

    def xeval_str(node):
        return node.s

    def xeval_attribute(node, ctx):
        assert type(node.value) == ast.Name
        t = node.value.id.capitalize()
        if t in [ 'Acquisition', 'Setup' ]:
            ctx.from_tables.update([ 'XImage', 'XImageParam AS Acquisition', 'XImageParam AS Setup' ])
            ctx.where_conjs.update([ 'Acquisition.param_type=0', 'Acquisition.ximage_id=XImage.id', 'Setup.param_type=1', 'Setup.ximage_id=XImage.id' ])
            ctx.where_conjs.add('%s.name=%s' % (t, ctx.push_param(node.attr)))
            return 'xvalue_parse(%s.val)' % (t,)
        elif t == 'Item':
            ctx.from_tables.update([ 'XImage', 'XBelonging', 'XBlob', 'XClass' ])
            ctx.where_conjs.update([ 'XImage.id=XBelonging.ximage_id', 'XBlob.xbelonging_id=XBelonging.id', 'XBlob.xclass_id=XClass.id' ])
            ctx.where_conjs.add('XClass.name=%s' % (ctx.push_param(node.attr),))
            return '*'
        else:
            pass # raise

    def xeval_call(node, ctx):
        fn = node.func.id.lower()
        if fn == 'count':
            assert len(node.args) == 1 and type(node.args[0]) == ast.Attribute
            return True, 'COUNT(%s)' % (xeval_attribute(node.args[0], ctx),)
        elif fn == 'area':
            assert len(node.args) == 1 and type(node.args[0]) == ast.Attribute
            xeval_attribute(node.args[0], ctx)
            return True, 'XBlob.area'
        elif fn == 'areas':
            assert len(node.args) == 1 and type(node.args[0]) == ast.Attribute
            xeval_attribute(node.args[0], ctx)
            return True, 'SUM(XBlob.area)'
        else:
            pass # Raise

    def xeval_unaryop(node, ctx):
        if type(node.op) == ast.Not:
            return ctx.all_paths - xeval(node.operand, cur)
        else:
            pass # Raise

    def xeval_boolop(node, ctx):
        values = [ xeval(v, ctx) for v in node.values ]
        if type(node.op) == ast.And:
            return reduce(set.intersection, values, ctx.all_paths)
        elif type(node.op) == ast.Or:
            return reduce(set.union, values, set())
        else:
            pass # Raise

    def xeval_compare(node, ctx):
        comparators = [ node.left ] + node.comparators
        paths = ctx.all_paths
        for op, x, y in zip(map(type, node.ops), comparators[:-1], comparators[1:]):
            if op == ast.Lt:
                op_str = '<'
            elif op == ast.LtE:
                op_str = '<='
            elif op == ast.Gt:
                op_str = '>'
            elif op == ast.GtE:
                op_str = '>='
            elif op == ast.Eq:
                op_str = '='
            elif op == ast.NotEq:
                op_str = '<>'
            else:
                pass # raise

            comps = [ '', '' ]
            conjs = ctx.where_conjs
            for i, z in enumerate([ x, y ]):
                if type(z) == ast.Call:
                    h, comps[i] = xeval_call(z, ctx)
                    if h:
                        conjs = ctx.having_conjs
                elif type(z) == ast.Attribute:
                    comps[i] = xeval_attribute(z, ctx)
                elif type(z) == ast.Str:
                    comps[i] = ctx.push_param(xeval_str(z))
                elif type(z) == ast.Num:
                    comps[i] = ctx.push_param(xeval_num(z))
                else:
                    pass # raise
            conjs.add('%s%s%s' % (comps[0], op_str, comps[1]))

            #
            paths = paths.intersection(ctx.execute_query())
            ctx.reset()
            if len(paths) == 0:
                break
        return paths

    def xeval(node, ctx):
        if type(node) == ast.UnaryOp:
            return xeval_unaryop(node, ctx)
        elif type(node) == ast.BoolOp:
            return xeval_boolop(node, ctx)
        elif type(node) == ast.Compare:
            return xeval_compare(node, ctx)
        else:
            pass # raise

    try:
        conn = _ximage_index_connect(args)
    except IOError as e:
        sys.stderr.write('Error: cannot open index: %s\n' % (str(e),))
        return 1
    except ImportError:
        sys.stderr.write('Error: cannot import sqlite3 module\n')
        return -1

    query = ' '.join(args.query)
    if query is None or len(query.strip()) == 0:
        paths = XEvalContext(conn.cursor()).all_paths
    else:
        root = ast.parse(query, '<query>', 'eval')
        paths = xeval(root.body, XEvalContext(conn.cursor()))
    print('\n'.join(sorted(paths)))
    return 0

def ximage_main(prog_name='ximage'):
    parser = argparse.ArgumentParser(prog=prog_name, description='Manipulate images along with its metadata')
    subparsers = parser.add_subparsers(help='sub-commands help')

    parser_import = subparsers.add_parser('import', help='Add blobs and metadata to an image, importing index mask')
    parser_import.add_argument('-K', '--classes', type=str, required=False, nargs='+', default=[], help='List of classes, 0-indexed')
    parser_import.add_argument('-U', '--uuids', type=str, required=False, nargs='+', default=[], help='List of UUIDs (0 to generate)')
    parser_import.add_argument('-C', '--colors', type=str, required=False, nargs='+', default=[], help='List of classes\' colors')
    parser_import.add_argument('mask', type=str, help='Index mask path')
    parser_import.add_argument('path', type=str, help='Image path')
    parser_import.set_defaults(func=ximage_import)

    parser_export = subparsers.add_parser('export', help='Export index mask from an image')
    parser_export.add_argument('path', type=str, help='Image path')
    parser_export.add_argument('mask', type=str, help='Index mask path')
    parser_export.set_defaults(func=ximage_export)

    parser_inject = subparsers.add_parser('inject', help='Add blobs and metadata to an image')
    parser_inject.add_argument('metadata', type=str, help='XML')
    parser_inject.add_argument('path', type=str, help='Image path')
    parser_inject.set_defaults(func=ximage_inject)

    parser_extract = subparsers.add_parser('extract', help='Extract blobs and metadata from an image')
    parser_extract.add_argument('path', type=str, help='Image path')
    parser_extract.set_defaults(func=ximage_extract)

    parser_update = subparsers.add_parser('update', help='Update image metadata with XML')
    parser_update.add_argument('-f', '--overwrite', action='store_true', required=False, default=False, help='Overwrite present values (default: no)')
    parser_update.add_argument('-K', '--replace-classes', action='store_true', required=False, default=False, help='Overwrite all defined classes (default: no)')
    parser_update.add_argument('metadata', type=str, help='Metadata to update with')
    parser_update.add_argument('path', type=str, help='Image path')
    parser_update.add_argument('mapping', nargs=argparse.REMAINDER)
    parser_update.set_defaults(func=ximage_update)

    parser_uuid = subparsers.add_parser('uuid', help='Get/set items UUIDs (left to right, top to bottom)')
    parser_uuid.add_argument('-U', '--uuids', type=str, required=False, nargs='+', default=[], help='List of new UUIDs (0 to skip)')
    parser_uuid.add_argument('path', type=str, help='Image path')
    parser_uuid.set_defaults(func=ximage_uuid)

    parser_view = subparsers.add_parser('view', help='View images, blobs and other metadata')
    parser_view.add_argument('-m', '--metadata', type=str, required=False, default=None, help='Use this XML instead of image\'s XMP')
    parser_view.add_argument('path', type=str, help='Image path')
    parser_view.set_defaults(func=ximage_view)

    parser_index = subparsers.add_parser('index', help='Index a directory (recursively) of XImages')
    parser_index.add_argument('root', type=str, help='Root directory path')
    parser_index.set_defaults(func=ximage_index)

    parser_query = subparsers.add_parser('query', help='Query on indexed directory of XImages')
    parser_query.add_argument('-D', '--root', type=str, required=False, default=os.getcwd(), help='Root directory path (default: cwd)')
    parser_query.add_argument('query', nargs=argparse.REMAINDER)
    parser_query.set_defaults(func=ximage_query)

    args = parser.parse_args()
    sys.exit(args.func(args))

_COLORS = dict(
    maroon=(0x00, 0x00, 0x80),
    darkred=(0x00, 0x00, 0x8b),
    red=(0x00, 0x00, 0xff),
    lightpink=(0xc1, 0xb6, 0xff),
    crimson=(0x3c, 0x14, 0xdc),
    palevioletred=(0x93, 0x70, 0xdb),
    hotpink=(0xb4, 0x69, 0xff),
    deeppink=(0x93, 0x14, 0xff),
    mediumvioletred=(0x85, 0x15, 0xc7),
    purple=(0x80, 0x00, 0x80),
    darkmagenta=(0x8b, 0x00, 0x8b),
    orchid=(0xd6, 0x70, 0xda),
    thistle=(0xd8, 0xbf, 0xd8),
    plum=(0xdd, 0xa0, 0xdd),
    violet=(0xee, 0x82, 0xee),
    fuchsia=(0xff, 0x00, 0xff),
    magenta=(0xff, 0x00, 0xff),
    mediumorchid=(0xd3, 0x55, 0xba),
    darkviolet=(0xd3, 0x00, 0x94),
    darkorchid=(0xcc, 0x32, 0x99),
    blueviolet=(0xe2, 0x2b, 0x8a),
    indigo=(0x82, 0x00, 0x4b),
    mediumpurple=(0xdb, 0x70, 0x93),
    slateblue=(0xcd, 0x5a, 0x6a),
    mediumslateblue=(0xee, 0x68, 0x7b),
    darkblue=(0x8b, 0x00, 0x00),
    mediumblue=(0xcd, 0x00, 0x00),
    blue=(0xff, 0x00, 0x00),
    navy=(0x80, 0x00, 0x00),
    midnightblue=(0x70, 0x19, 0x19),
    darkslateblue=(0x8b, 0x3d, 0x48),
    royalblue=(0xe1, 0x69, 0x41),
    cornflowerblue=(0xed, 0x95, 0x64),
    lightsteelblue=(0xde, 0xc4, 0xb0),
    aliceblue=(0xff, 0xf8, 0xf0),
    ghostwhite=(0xff, 0xf8, 0xf8),
    lavender=(0xfa, 0xe6, 0xe6),
    dodgerblue=(0xff, 0x90, 0x1e),
    steelblue=(0xb4, 0x82, 0x46),
    deepskyblue=(0xff, 0xbf, 0x00),
    slategray=(0x90, 0x80, 0x70),
    lightslategray=(0x99, 0x88, 0x77),
    lightskyblue=(0xfa, 0xce, 0x87),
    skyblue=(0xeb, 0xce, 0x87),
    lightblue=(0xe6, 0xd8, 0xad),
    teal=(0x80, 0x80, 0x00),
    darkcyan=(0x8b, 0x8b, 0x00),
    darkturquoise=(0xd1, 0xce, 0x00),
    cyan=(0xff, 0xff, 0x00),
    mediumturquoise=(0xcc, 0xd1, 0x48),
    cadetblue=(0xa0, 0x9e, 0x5f),
    paleturquoise=(0xee, 0xee, 0xaf),
    lightcyan=(0xff, 0xff, 0xe0),
    azure=(0xff, 0xff, 0xf0),
    lightseagreen=(0xaa, 0xb2, 0x20),
    turquoise=(0xd0, 0xe0, 0x40),
    powderblue=(0xe6, 0xe0, 0xb0),
    darkslategray=(0x4f, 0x4f, 0x2f),
    aquamarine=(0xd4, 0xff, 0x7f),
    mediumspringgreen=(0x9a, 0xfa, 0x00),
    mediumaquamarine=(0xaa, 0xcd, 0x66),
    springgreen=(0x7f, 0xff, 0x00),
    mediumseagreen=(0x71, 0xb3, 0x3c),
    seagreen=(0x57, 0x8b, 0x2e),
    limegreen=(0x32, 0xcd, 0x32),
    darkgreen=(0x00, 0x64, 0x00),
    green=(0x00, 0x80, 0x00),
    lime=(0x00, 0xff, 0x00),
    forestgreen=(0x22, 0x8b, 0x22),
    darkseagreen=(0x8f, 0xbc, 0x8f),
    lightgreen=(0x90, 0xee, 0x90),
    palegreen=(0x98, 0xfb, 0x98),
    mintcream=(0xfa, 0xff, 0xf5),
    honeydew=(0xf0, 0xff, 0xf0),
    chartreuse=(0x00, 0xff, 0x7f),
    lawngreen=(0x00, 0xfc, 0x7c),
    olivedrab=(0x23, 0x8e, 0x6b),
    darkolivegreen=(0x2f, 0x6b, 0x55),
    yellowgreen=(0x32, 0xcd, 0x9a),
    greenyellow=(0x2f, 0xff, 0xad),
    beige=(0xdc, 0xf5, 0xf5),
    linen=(0xe6, 0xf0, 0xfa),
    lightgoldenrodyellow=(0xd2, 0xfa, 0xfa),
    olive=(0x00, 0x80, 0x80),
    yellow=(0x00, 0xff, 0xff),
    lightyellow=(0xe0, 0xff, 0xff),
    ivory=(0xf0, 0xff, 0xff),
    darkkhaki=(0x6b, 0xb7, 0xbd),
    khaki=(0x8c, 0xe6, 0xf0),
    palegoldenrod=(0xaa, 0xe8, 0xee),
    wheat=(0xb3, 0xde, 0xf5),
    gold=(0x00, 0xd7, 0xff),
    lemonchiffon=(0xcd, 0xfa, 0xff),
    papayawhip=(0xd5, 0xef, 0xff),
    darkgoldenrod=(0x0b, 0x86, 0xb8),
    goldenrod=(0x20, 0xa5, 0xda),
    antiquewhite=(0xd7, 0xeb, 0xfa),
    cornsilk=(0xdc, 0xf8, 0xff),
    oldlace=(0xe6, 0xf5, 0xfd),
    moccasin=(0xb5, 0xe4, 0xff),
    navajowhite=(0xad, 0xde, 0xff),
    orange=(0x00, 0xa5, 0xff),
    bisque=(0xc4, 0xe4, 0xff),
    tan=(0x8c, 0xb4, 0xd2),
    darkorange=(0x00, 0x8c, 0xff),
    burlywood=(0x87, 0xb8, 0xde),
    saddlebrown=(0x13, 0x45, 0x8b),
    sandybrown=(0x60, 0xa4, 0xf4),
    blanchedalmond=(0xcd, 0xeb, 0xff),
    lavenderblush=(0xf5, 0xf0, 0xff),
    seashell=(0xee, 0xf5, 0xff),
    floralwhite=(0xf0, 0xfa, 0xff),
    snow=(0xfa, 0xfa, 0xff),
    peru=(0x3f, 0x85, 0xcd),
    peachpuff=(0xb9, 0xda, 0xff),
    chocolate=(0x1e, 0x69, 0xd2),
    sienna=(0x2d, 0x52, 0xa0),
    lightsalmon=(0x7a, 0xa0, 0xff),
    coral=(0x50, 0x7f, 0xff),
    darksalmon=(0x7a, 0x96, 0xe9),
    mistyrose=(0xe1, 0xe4, 0xff),
    orangered=(0x00, 0x45, 0xff),
    salmon=(0x72, 0x80, 0xfa),
    tomato=(0x47, 0x63, 0xff),
    rosybrown=(0x8f, 0x8f, 0xbc),
    pink=(0xcb, 0xc0, 0xff),
    indianred=(0x5c, 0x5c, 0xcd),
    lightcoral=(0x80, 0x80, 0xf0),
    brown=(0x2a, 0x2a, 0xa5),
    firebrick=(0x22, 0x22, 0xb2),
    black=(0x00, 0x00, 0x00),
    dimgray=(0x69, 0x69, 0x69),
    gray=(0x80, 0x80, 0x80),
    darkgray=(0xa9, 0xa9, 0xa9),
    silver=(0xc0, 0xc0, 0xc0),
    lightgrey=(0xd3, 0xd3, 0xd3),
    gainsboro=(0xdc, 0xdc, 0xdc),
    whitesmoke=(0xf5, 0xf5, 0xf5),
    white=(0xff, 0xff, 0xff)
)

_XIMAGE_INDEX_CREATE_SCHEMA = """
-- Parse::SQL::Dia       version 0.27
-- Documentation         http://search.cpan.org/dist/Parse-Dia-SQL/
-- Environment           Perl 5.018002, /usr/bin/perl
-- Architecture          x86_64-linux-gnu-thread-multi
-- Target Database       sqlite3fk
-- Input file            ximage_schema.dia
-- Generated at          Tue Sep 19 16:21:30 2017
-- Typemap for sqlite3fk not found in input file

-- get_constraints_drop
drop index if exists idx_xccnc;
drop index if exists idx_ximxit;

-- get_permissions_drop

-- get_view_drop

-- get_schema_drop
drop table if exists XBlob;
drop table if exists XItem;
drop table if exists XImage;
drop table if exists XClass;
drop table if exists XImageParam;
drop table if exists XBelonging;

-- get_smallpackage_pre_sql

-- get_schema_create

create table XBlob (
   id            integer not null,
   xbelonging_id integer not null,
   parent_id     integer null    ,
   xclass_id     integer not null,
   val           real    not null,
   area          real    not null,
   vals          vector  not null,
   contour       points  not null,
   constraint pk_XBlob primary key (id),
   foreign key(xclass_id) references XClass(id) ,
   foreign key(parent_id) references XBlob(id) ,
   foreign key(xbelonging_id) references XBelonging(id)
)   ;

create table XItem (
   id uuid not null,
   constraint pk_XItem primary key (id)
)   ;

create table XImage (
   id   uuid not null,
   path text not null,
   constraint pk_XImage primary key (id)
)   ;

create table XClass (
   id      integer not null,
   classid integer not null,
   name    string  not null,
   color   color   not null,
   constraint pk_XClass primary key (id)
)   ;

create table XImageParam (
   ximage_id  uuid    not null,
   param_type integer not null,
   name       text    not null,
   val        xvalue  not null,
   constraint pk_XImageParam primary key (ximage_id,param_type,name),
   foreign key(ximage_id) references XImage(id)
)   ;

create table XBelonging (
   id        integer not null,
   ximage_id uuid    not null,
   xitem_id  uuid    not null,
   constraint pk_XBelonging primary key (id),
   foreign key(ximage_id) references XImage(id) ,
   foreign key(xitem_id) references XItem(id)
)   ;

-- get_view_create

-- get_permissions_create

-- get_inserts

-- get_smallpackage_post_sql

-- get_associations_create
create unique index idx_xccnc on XClass (classid,name,color) ;
create unique index idx_ximxit on XBelonging (ximage_id,xitem_id) ;
"""

if __name__ == '__main__':
    ximage_main()
