from base_config import BaseConfig

class StreamConfig(BaseConfig):
    # Server configs
    service = {
        "host" : "0.0.0.0",
        "port" : "54443",
        "ZMQ_URL_CR_CL" : "tcp://localhost:54321",
        "ZMQ_URL_FACE" : "tcp://localhost:54444",
        "STREAM_URL" : "rtmp://0.0.0.0:1935/live",
        "STREAM_SERVER" : "nginx", #nginx or wowza
        "ffmpeg_path" : "/home/dogus/ffmpeg_install/FFmpeg/ffmpeg",
        "gpu_to_use" : "1"
    }

    # Available actions
    actions = {
        "ACTION_START" : 0,
        "ACTION_STOP" : 1,
        "ACTION_CHECK" : 2
    }

    # Stream-related configs
    stream = {
        "TYPE_CAR_CLASSIFICATION" : 2,
        "TYPE_FACE_DETECTION" : 1,
        "INTERVAL" : 12,
        "COPY_COUNT" : 10,
        "RECONNECT_TIME_OUT" : 5,
        "RECONNECT_TRY_COUNT" : 5
    }

    # Configs related to check stream status
    wowza_stream_stat = {
        "auth-user" : "dogus",
        "auth-pass" : "ares2728",
        "url" : "http://localhost:8087/v2/servers/_defaultServer_/vhosts/_defaultVHost_/applications/live/instances/_definst_",
        "headers" : {
            'Accept': 'application/json; charset=utf-8',
        }
    }

    nginx_stream_stat = {
        "url" : "http://localhost:8888/stats",
    }

    ipcam_demo = {
        "in_demo_mode" : False,
        "recog_save_path" : "/var/stream_files/face_logs",
        "timestamp_format" : "%Y-%m-%d_%H:%M:%S",
    }
