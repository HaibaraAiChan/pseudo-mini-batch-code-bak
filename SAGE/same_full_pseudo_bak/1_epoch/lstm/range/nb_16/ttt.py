filename='./tt.log'
filename='./2_layer.log'
res=[]
with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("torch.Size("):
				res.append(int(line.split(',')[1]))
print(sum(res))