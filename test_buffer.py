from replay_buffer_depth import ReplayBufferDepth


size= 256
memory = ReplayBufferDepth((size, size), (size,size,3), (size, size, 3), 15000, "cuda")
#memory.load_memory("depth_memory5k")
memory.load_memory("depth_memory10k")

memory.create_surface_normals("normals_memory10k")

print(memory.idx)
# memory.test_surface_normals(6423)
memory.test_surface_normals(49)
