// Auto-generated. Do not edit!

// (in-package orbbec_camera.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class StreamProfile {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.stream_name = null;
      this.width = null;
      this.height = null;
      this.fps = null;
      this.format = null;
    }
    else {
      if (initObj.hasOwnProperty('stream_name')) {
        this.stream_name = initObj.stream_name
      }
      else {
        this.stream_name = '';
      }
      if (initObj.hasOwnProperty('width')) {
        this.width = initObj.width
      }
      else {
        this.width = 0;
      }
      if (initObj.hasOwnProperty('height')) {
        this.height = initObj.height
      }
      else {
        this.height = 0;
      }
      if (initObj.hasOwnProperty('fps')) {
        this.fps = initObj.fps
      }
      else {
        this.fps = 0;
      }
      if (initObj.hasOwnProperty('format')) {
        this.format = initObj.format
      }
      else {
        this.format = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type StreamProfile
    // Serialize message field [stream_name]
    bufferOffset = _serializer.string(obj.stream_name, buffer, bufferOffset);
    // Serialize message field [width]
    bufferOffset = _serializer.int32(obj.width, buffer, bufferOffset);
    // Serialize message field [height]
    bufferOffset = _serializer.int32(obj.height, buffer, bufferOffset);
    // Serialize message field [fps]
    bufferOffset = _serializer.int32(obj.fps, buffer, bufferOffset);
    // Serialize message field [format]
    bufferOffset = _serializer.string(obj.format, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type StreamProfile
    let len;
    let data = new StreamProfile(null);
    // Deserialize message field [stream_name]
    data.stream_name = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [width]
    data.width = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [height]
    data.height = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [fps]
    data.fps = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [format]
    data.format = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.stream_name);
    length += _getByteLength(object.format);
    return length + 20;
  }

  static datatype() {
    // Returns string type for a message object
    return 'orbbec_camera/StreamProfile';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'eab518462680efdcb8df71e9c4ed57d9';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string stream_name
    int32 width
    int32 height
    int32 fps
    string format
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new StreamProfile(null);
    if (msg.stream_name !== undefined) {
      resolved.stream_name = msg.stream_name;
    }
    else {
      resolved.stream_name = ''
    }

    if (msg.width !== undefined) {
      resolved.width = msg.width;
    }
    else {
      resolved.width = 0
    }

    if (msg.height !== undefined) {
      resolved.height = msg.height;
    }
    else {
      resolved.height = 0
    }

    if (msg.fps !== undefined) {
      resolved.fps = msg.fps;
    }
    else {
      resolved.fps = 0
    }

    if (msg.format !== undefined) {
      resolved.format = msg.format;
    }
    else {
      resolved.format = ''
    }

    return resolved;
    }
};

module.exports = StreamProfile;
