"""Shared video writer."""
def save_mp4(frames, path: str, fps: int = 30) -> None:
    import av
    h, w = frames[0].shape[:2]
    with av.open(path, "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        for f in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
