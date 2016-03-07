import graphlab as gl
import _init__
import ipdb
import tools.rpn_matching as match
# para inint
topk = 5
# +++++load data, select query: cls, qid
data = gl.load_sframe("./data_237.gl")
cls = list(set(data["cls"]))
qid = 0
query = data["q_img"][qid]
ipdb.set_trace()
# +++++RPN, get rois; nn top k
net_default = "vgg16"
neighbors, db_sf, cand_sf = match.demo(net_default, query, db)
# +++++join table, .show()
neighbors = neighbors.add_row_number()
neighbors.print_rows()
# check para seq
match.image_join(qqid, db_sf, cand_sf)

def get_topk_match(neighbors, topk=5):
  neighbors["score"]
  return topk_rois, topk_cand

def join(topk_cand, cata_GT):
