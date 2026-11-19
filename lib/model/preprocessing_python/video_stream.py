"""Select the moving-picture stream consistently across video backends."""


def select_video_stream(container):
    """Return the first video stream that is not cover art or thumbnails.

    Embedded images are also exposed through ``streams.video`` and can precede
    the actual video, notably with ASF/WMV cover art. Keep the first-stream
    policy for ordinary video tracks, but never analyze an attached image as
    if it were the movie.
    """
    from av.stream import Disposition

    images = Disposition.attached_pic | Disposition.timed_thumbnails
    for stream in container.streams.video:
        if not stream.disposition & images:
            return stream
    raise ValueError("No video stream other than attached pictures or thumbnails")
