// Auto-generated. Do not edit!

// (in-package orbbec_camera.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let StreamProfile = require('../msg/StreamProfile.js');

//-----------------------------------------------------------


//-----------------------------------------------------------

class SetStreamProfileRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.profiles = null;
    }
    else {
      if (initObj.hasOwnProperty('profiles')) {
        this.profiles = initObj.profiles
      }
      else {
        this.profiles = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SetStreamProfileRequest
    // Serialize message field [profiles]
    // Serialize the length for message field [profiles]
    bufferOffset = _serializer.uint32(obj.profiles.length, buffer, bufferOffset);
    obj.profiles.forEach((val) => {
      bufferOffset = StreamProfile.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetStreamProfileRequest
    let len;
    let data = new SetStreamProfileRequest(null);
    // Deserialize message field [profiles]
    // Deserialize array length for message field [profiles]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.profiles = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.profiles[i] = StreamProfile.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.profiles.forEach((val) => {
      length += StreamProfile.getMessageSize(val);
    });
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'orbbec_camera/SetStreamProfileRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '0df140e3681dce7d8ca58c5863dad9eb';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    StreamProfile[] profiles
    
    ================================================================================
    MSG: orbbec_camera/StreamProfile
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
    const resolved = new SetStreamProfileRequest(null);
    if (msg.profiles !== undefined) {
      resolved.profiles = new Array(msg.profiles.length);
      for (let i = 0; i < resolved.profiles.length; ++i) {
        resolved.profiles[i] = StreamProfile.Resolve(msg.profiles[i]);
      }
    }
    else {
      resolved.profiles = []
    }

    return resolved;
    }
};

class SetStreamProfileResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.success = null;
      this.message = null;
    }
    else {
      if (initObj.hasOwnProperty('success')) {
        this.success = initObj.success
      }
      else {
        this.success = false;
      }
      if (initObj.hasOwnProperty('message')) {
        this.message = initObj.message
      }
      else {
        this.message = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SetStreamProfileResponse
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    // Serialize message field [message]
    bufferOffset = _serializer.string(obj.message, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetStreamProfileResponse
    let len;
    let data = new SetStreamProfileResponse(null);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [message]
    data.message = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.message);
    return length + 5;
  }

  static datatype() {
    // Returns string type for a service object
    return 'orbbec_camera/SetStreamProfileResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '937c9679a518e3a18d831e57125ea522';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool success
    string message
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SetStreamProfileResponse(null);
    if (msg.success !== undefined) {
      resolved.success = msg.success;
    }
    else {
      resolved.success = false
    }

    if (msg.message !== undefined) {
      resolved.message = msg.message;
    }
    else {
      resolved.message = ''
    }

    return resolved;
    }
};

module.exports = {
  Request: SetStreamProfileRequest,
  Response: SetStreamProfileResponse,
  md5sum() { return 'd2095f22bd3c757e798fc2eece95a412'; },
  datatype() { return 'orbbec_camera/SetStreamProfile'; }
};
