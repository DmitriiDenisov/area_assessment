import numpy as np
import cv2
from collections import defaultdict
from shapely.geometry import Polygon, MultiPolygon
from descartes import PolygonPatch
from gdal import ogr, osr
import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection


def mask_to_polygons(mask, epsilon=5, min_area=.1, rect_polygon=False):
    horiz_axis = float(mask.shape[0] - 1) / 2
    vert_axis = float(mask.shape[1] - 1) / 2
    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            x_coord = cnt[:, 0, 0]
            y_coord = cnt[:, 0, 1]
            cnt[:, 0, 1] = 2 * vert_axis - y_coord
            if rect_polygon:
                cnt = cv2.boxPoints(cv2.minAreaRect(cnt))  # rectangular polygons
                poly = Polygon(
                    shell=cnt[:, :],
                    holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                           if cv2.contourArea(c) >= min_area])
            else:
                poly = Polygon(
                    shell=cnt[:, 0, :],
                    holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                           if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


def plot_polygons(mp, pname, output_folder):

    cm = plt.get_cmap('RdBu')
    num_colours = len(mp)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    minx, miny, maxx, maxy = mp.bounds
    # w, h = maxx - minx, maxy - miny
    # ax.set_xlim(minx - 0.2 * w, maxx + 0.2 * w)
    # ax.set_ylim(miny - 0.2 * h, maxy + 0.2 * h)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect(1)

    patches = []
    for idx, p in enumerate(mp):
        colour = cm(1. * idx / num_colours)
        patches.append(PolygonPatch(p, fc=colour, ec='#555555', lw=0.2, alpha=1., zorder=1))
    ax.add_collection(PatchCollection(patches, match_original=True))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Shapely polygons rendered using Shapely")
    plt.tight_layout()
    plt.savefig(output_folder+'{}'.format(pname), alpha=True, dpi=500)
    plt.show()


def save_polygons(poly, output_folder, fname, meta=None):

    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(output_folder+'{}.shp'.format(fname))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    layer = ds.CreateLayer('', srs, ogr.wkbMultiPolygon)

    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    ## If there are multiple geometries, put the "for" loop here

    # Create a new feature (attribute and geometry)
    feat = ogr.Feature(defn)
    feat.SetField('id', 123)

    # Make a geometry, from Shapely object
    geom = ogr.CreateGeometryFromWkb(poly.wkb)
    feat.SetGeometry(geom)

    layer.CreateFeature(feat)

    feat = geom = None  # destroy these

    # Save and close everything
    ds = layer = feat = geom = None


def contours_to_polygons(contour_list, shapes, tolerance=0.2):
    n = len(contour_list)
    polygon_list = list(range(n))
    for i in range(n):
        print('Polygons from contours. Image #{i} out of {n}'.format(i=i + 1, n=n))
        polygon_list[i] = []
        vert_axis = float(shapes[i][1] - 1) / 2
        for cnt in contour_list[i]:
            y_coord = cnt[:, 0, 1]
            cnt[:, 0, 1] = 2 * vert_axis - y_coord
            try:
                if tolerance > 0:
                    polygon_list[i].append(Polygon(shell=cnt[:, 0, :]).simplify(tolerance, preserve_topology=True))
                else:
                    polygon_list[i].append(Polygon(shell=cnt[:, 0, :]))
            except ValueError:
                i = i
        # approximating polygons might have created invalid ones, fix them
        polygon_list[i] = MultiPolygon(polygon_list[i])
        if not polygon_list[i].is_valid:
            polygon_list[i] = polygon_list[i].buffer(0)
            # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
            # need to keep it a Multi throughout
            if polygon_list[i].type == 'Polygon':
                polygon_list[i] = MultiPolygon([polygon_list[i]])
    return polygon_list

