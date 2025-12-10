from pylsl import StreamInfo, StreamOutlet


def create_lsl_outlet(stream_name="GoNoGoMarkers", stream_type="Markers", channel_count=1, source_id="gonogo_stream"):
    """
    Initialize an LSL stream for event markers.
    Returns a StreamOutlet object.
    """
    info = StreamInfo(
        name=stream_name,
        type=stream_type,
        channel_count=channel_count,
        nominal_srate=0,          # irregular stream
        channel_format="string",  # we send text markers
        source_id=source_id
    )
    outlet = StreamOutlet(info)
    print(f"LSL stream initialized: {stream_name} ({stream_type})")
    
    return outlet


def send_marker(outlet, marker):
    """Send a marker string via LSL."""
    if outlet:
        outlet.push_sample([marker])