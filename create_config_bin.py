import struct

values = [768, 3072, 12, 12, 12, 50257, 1024]
with open("config.bin", "wb") as f:
    for v in values:
        f.write(struct.pack("<i", v))  # little-endian int32
