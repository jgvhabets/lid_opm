from pylsl import StreamInfo, StreamOutlet

def create_lsl_outlet(stream_name="GoNoGoMarkers", stream_type="Markers"):
    """Create and return an LSL outlet for sending string markers."""
    info = StreamInfo(name=stream_name, type=stream_type, channel_count=1,
                      nominal_srate=0, channel_format='string', source_id="gng_task_01")
    outlet = StreamOutlet(info)
    return outlet

def send_marker(outlet, marker):
    """Send a marker string via LSL."""
    if outlet:
        outlet.push_sample([marker])