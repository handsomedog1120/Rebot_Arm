; Auto-generated. Do not edit!


(cl:in-package orbbec_camera-msg)


;//! \htmlinclude StreamProfile.msg.html

(cl:defclass <StreamProfile> (roslisp-msg-protocol:ros-message)
  ((stream_name
    :reader stream_name
    :initarg :stream_name
    :type cl:string
    :initform "")
   (width
    :reader width
    :initarg :width
    :type cl:integer
    :initform 0)
   (height
    :reader height
    :initarg :height
    :type cl:integer
    :initform 0)
   (fps
    :reader fps
    :initarg :fps
    :type cl:integer
    :initform 0)
   (format
    :reader format
    :initarg :format
    :type cl:string
    :initform ""))
)

(cl:defclass StreamProfile (<StreamProfile>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <StreamProfile>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'StreamProfile)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name orbbec_camera-msg:<StreamProfile> is deprecated: use orbbec_camera-msg:StreamProfile instead.")))

(cl:ensure-generic-function 'stream_name-val :lambda-list '(m))
(cl:defmethod stream_name-val ((m <StreamProfile>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader orbbec_camera-msg:stream_name-val is deprecated.  Use orbbec_camera-msg:stream_name instead.")
  (stream_name m))

(cl:ensure-generic-function 'width-val :lambda-list '(m))
(cl:defmethod width-val ((m <StreamProfile>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader orbbec_camera-msg:width-val is deprecated.  Use orbbec_camera-msg:width instead.")
  (width m))

(cl:ensure-generic-function 'height-val :lambda-list '(m))
(cl:defmethod height-val ((m <StreamProfile>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader orbbec_camera-msg:height-val is deprecated.  Use orbbec_camera-msg:height instead.")
  (height m))

(cl:ensure-generic-function 'fps-val :lambda-list '(m))
(cl:defmethod fps-val ((m <StreamProfile>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader orbbec_camera-msg:fps-val is deprecated.  Use orbbec_camera-msg:fps instead.")
  (fps m))

(cl:ensure-generic-function 'format-val :lambda-list '(m))
(cl:defmethod format-val ((m <StreamProfile>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader orbbec_camera-msg:format-val is deprecated.  Use orbbec_camera-msg:format instead.")
  (format m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <StreamProfile>) ostream)
  "Serializes a message object of type '<StreamProfile>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'stream_name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'stream_name))
  (cl:let* ((signed (cl:slot-value msg 'width)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'height)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'fps)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'format))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'format))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <StreamProfile>) istream)
  "Deserializes a message object of type '<StreamProfile>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'stream_name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'stream_name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'width) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'height) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'fps) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'format) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'format) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<StreamProfile>)))
  "Returns string type for a message object of type '<StreamProfile>"
  "orbbec_camera/StreamProfile")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'StreamProfile)))
  "Returns string type for a message object of type 'StreamProfile"
  "orbbec_camera/StreamProfile")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<StreamProfile>)))
  "Returns md5sum for a message object of type '<StreamProfile>"
  "eab518462680efdcb8df71e9c4ed57d9")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'StreamProfile)))
  "Returns md5sum for a message object of type 'StreamProfile"
  "eab518462680efdcb8df71e9c4ed57d9")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<StreamProfile>)))
  "Returns full string definition for message of type '<StreamProfile>"
  (cl:format cl:nil "string stream_name~%int32 width~%int32 height~%int32 fps~%string format~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'StreamProfile)))
  "Returns full string definition for message of type 'StreamProfile"
  (cl:format cl:nil "string stream_name~%int32 width~%int32 height~%int32 fps~%string format~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <StreamProfile>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'stream_name))
     4
     4
     4
     4 (cl:length (cl:slot-value msg 'format))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <StreamProfile>))
  "Converts a ROS message object to a list"
  (cl:list 'StreamProfile
    (cl:cons ':stream_name (stream_name msg))
    (cl:cons ':width (width msg))
    (cl:cons ':height (height msg))
    (cl:cons ':fps (fps msg))
    (cl:cons ':format (format msg))
))
