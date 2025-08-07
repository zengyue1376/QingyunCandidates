
import pickle
path='F:\\Work\\202508_TencentAd\\TencentGR_1k\\TencentGR_1k\\indexer.pkl'   #path='/root/……/aus_openface.pkl'   pkl文件所在路径
	   
f=open(path,'rb')
data=pickle.load(f)
 
# print(len(data))
# print(type(data))
# print(data.keys())
# print(type(data["f"]))
# print(len(data['f']))
# print(data["f"].keys())
# # print(data["f"]["122"])
# print(len(data["u"].keys()))
print(data["u"]["user_00997341"])
# print(len(data["i"].keys()))
# print(data["i"].keys()[0])
# print(data["i"][20002428751])

with open('F:\\Work\\202508_TencentAd\\TencentGR_1k\\TencentGR_1k\seq_offsets.pkl', 'rb') as f:
    seq_offsets = pickle.load(f)

print(seq_offsets, len(seq_offsets))