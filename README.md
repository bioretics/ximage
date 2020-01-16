# XImage

The eXtended image (Ximage) is a specification format that allows to embed
complex metadata into image files like JPEGs or PNGs. It's based on the XMP ISO
standard, an evolution of Exif.

The basic idea is to join the informations coming from the acquisition process
and from the hand-made labeling (or from an AI classification) with the actual
pixel data, in the same file, avoiding the use of sidecar files.

XImage use its own namespace (http://bioretics.com/aliquis) to define the
schema of the metadata, mainly composed of:

- Acquisition: a set of properties related to the acquision process. E.g: date,
progressive shot id, dataset name, ...
- Setup: a set of properties related to the acquisition setup. E.g: camera
parameters, lights configuration, ...
- Classes: an array of description-color pairs, describing what kind objects are
used in the annotations and how they must be displayed.
- Items: a set of objects present in the scene, identified by an UUID and
represented by a hierarchy of blobs, that are areas of pixels belonging to a
certain class.

The data are accessed through a single Python script `ximage.py`, which
implements both the library and the main manipulation tool.

## Functionalities

### ``extract`` and ``inject``

``extract`` reads the XML of a Ximage and save it to a file. ``injects``
performs the opposite operation, reading an XML file and embedding its content
into an existing image.
This last operation is tipically useful when preparing a large set of images
with the same base metadata, e.g. to inject acquisition, setup and classes
properties into a set of images belonging to the same acquisition.

### ``export`` and ``import``

These commands handle the conversion of Ximage's items to and from an index mask
(a grayscale image where each gray level corresponds to a class identifier).

With ``export`` it's possible to generate an index mask from a Ximage,
whereas ``import`` searches for contours in an index mask and save them as items
and blobs into a Ximage.
These tools are useful when there is the need to convert an old dataset of
image/index mask pairs to the Ximage format or, conversely, to create sibling
index masks for tools that don't support the Ximage format.

### ``update``, ``uuid`` and ``view``

``update`` command inject new metadata in an image that is already a Ximage (by
default, it will not overwrite or delete any current information). ``uuid``
is useful to get or set all items' UUID of a Ximage; ``view`` display items
and blobs in a graphical interface, by drawing contours of blobs whose colors
are defined by the respective class.

### ``index`` and ``query``

With these tools it's possibile to manage a large database of Ximages.

``index`` command reads all images inside a folder and for each Ximage it saves
metadata into a sqlite database, essentially indexing the whole folder, while
``query`` allows to search in the metadata database, and print the paths
of Ximages that match the required features.

#### Example of database creation and querying

Suppose we have a folder ``~/imgs`` of annotated Ximages of cats and dogs. We
can index the folder and create a database with:
```
  cd ~/imgs
  ximage.py index .
```

Then we can search for images that contains a cat with:
```
  ximage.py query 'count(item.cat) > 0'
```

This will result in a list of all images paths that contains at least a cat in
their annotation. Let's say we want images were there are both cats and dogs:
```
  ximage.py query 'count(item.cat) > 0 and count(item.dog) > 0'
```

If we want images containing at least 2 dogs:
```
  ximage.py query 'count(item.dog) >= 2'
```

The list of paths can than be used as an input for other programs, for example
to build a dataset of relevant images.

## Requirements

The tool is based on `OpenCV (>=3)` and `python-xmp-toolkit` (which in turn
require the Exempi library). It works with both Python 2 and Python 3.
