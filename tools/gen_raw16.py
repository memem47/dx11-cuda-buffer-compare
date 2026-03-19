from pathlib import Path
import struct
import numpy as np
import matplotlib.pyplot as plt

W = 1000
H = 700
OUT = Path("../data/input_1000x700_u16.raw")

OUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUT, "wb") as f:
    for y in range(H):
        for x in range(W):
            v = ((x * 37 + y * 91) ^ ((x // 8) << 4) ^ ((y // 8) << 3)) & 0xFFFF
            f.write(struct.pack("<H", v))

print(f"generated: {OUT}")
print(f"size = {W} x {H}, bytes = {W*H*2}")

# Display the generated image
with open(OUT, "rb") as f:
    raw_data = f.read()

# Convert raw bytes to numpy array
img_array = np.frombuffer(raw_data, dtype=np.uint16).reshape((H, W))

# Display using matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(img_array, cmap='gray', interpolation='nearest')
plt.colorbar(label='Pixel Value (16-bit)')
plt.title(f'Generated Image: {W}x{H}')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.show()