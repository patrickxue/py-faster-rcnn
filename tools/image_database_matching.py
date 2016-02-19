import graphlab as gl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
from skimage import io, transform
import time
import scipy
# gl.canvas.set_target('ipynb')

#image_path = '../data/demo/imgs/living_room.jpg' 

#get_ipython().magic(u'matplotlib inline')

#img = io.imread(image_path)


def get_candidates(img, scale=1000, sigma=2, min_size=50):
    t = time.time()
    img_lbl, regions = selectivesearch.selective_search(
                                                        img, 
                                                        scale=scale, 
                                                        sigma=sigma, 
                                                        min_size=min_size)
    print str(time.time() -t) + "s"
    candidates = set()

    for r in regions:
        x, y, w, h = r['rect']
            # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
            # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
            # distorted rects
        if w / h > 1.2 or h / w > 1.2:
            continue

        candidates.add(r['rect'])
        
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 12))
    ax.imshow(img)

    for x, y, w, h in candidates:
        rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1)

        ax.add_patch(rect)
        
    return candidates

#candidates = get_candidates(img)

def save_crops(img, candidates):
    count=0
      
    for x, y, w, h in candidates:    
        cropped = img[y:y+h, x:x+w, :]
        scipy.misc.imsave('crop_%d.jpg'%count, cropped)
        count+=1
        
#save_crops(image_path, candidates)

#cand_sf = gl.image_analysis.load_images('./')
#cand_sf['image'].show()

#dfe = gl.feature_engineering.DeepFeatureExtractor('image', model='auto', output_column_prefix=feat)
#dfe = gl.load_model('./alexnet.gl')

def transform_and_build_nn(cand_sf, dfe, radius=0.51, k=2):   
    cand_sf = dfe.transform(cand_sf)
    cand_sf = cand_sf.add_row_number()
    db_sf = gl.SFrame('/Users/charlie/Downloads/features_sframe.gl')
    db_sf = db_sf.add_row_number()
    nn = gl.nearest_neighbors.create(db_sf,features=['deep_features.image'],distance='cosine')
    neighbors = nn.query(cand_sf,radius=radius,k=k)
    return neighbors, db_sf, cand_sf

#neighbors, db_sf, cand_sf = transform_and_build_nn(cand_sf, .51, 2)

def image_join(neighbors, db_sf, cand_sf, query_id):
    tmp_nn = neighbors[neighbors['query_label'] == query_id]
    tmp_db = db_sf.filter_by(tmp_nn['reference_label'], 'id')
    cand = cand_sf[cand_sf['id'] == query_id]
    return cand.append(tmp_db)

#image_join(neighbors, db_sf, cand_sf, 9)['image'].show()
