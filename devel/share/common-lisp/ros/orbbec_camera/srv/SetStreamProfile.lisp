; Auto-generated. Do not edit!


(cl:in-package orbbec_camera-srv)


;//! \htmlinclude SetStreamProfile-request.msg.html

(cl:defclass <SetStreamProfile-request> (roslisp-msg-protocol:ros-message)
  ((profiles
    :reader profiles
    :initarg :profiles
    :type (cl:vector orbbec_camera-msg:StreamProfile)
   :initform (cl:make-array 0 :element-type 'orbbec_camera-msg:StreamProfile :initial-element (cl:make-instance 'orbbec_camera-msg:StreamProfile))))
)

(cl:defclass SetStreamProfile-request (<SetStreamProfile-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetStreamProfile-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetStreamProfile-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name orbbec_camera-srv:<SetStreamProfile-request> is deprecated: use orbbec_camera-srv:SetStreamProfile-request instead.")))

(cl:ensure-generic-function 'profiles-val :lambda-list '(m))
(cl:defmethod profiles-val ((m <SetStreamProfile-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader orbbec_camera-srv:profiles-val is deprecated.  Use orbbec_camera-srv:profiles instead.")
  (profiles m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetStreamProfile-request>) ostream)
  "Serializes a message object of type '<SetStreamProfile-request>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'profiles))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'profiles))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetStreamProfile-request>) istream)
  "Deserializes a message object of type '<SetStreamProfile-request>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'profiles) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'profiles)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'orbbec_camera-msg:StreamProfile))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetStreamProfile-request>)))
  "Returns string type for a service object of type '<SetStreamProfile-request>"
  "orbbec_camera/SetStreamProfileRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetStreamProfile-request)))
  "Returns string type for a service object of type 'SetStreamProfile-request"
  "orbbec_camera/SetStreamProfileRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetStreamProfile-request>)))
  "Returns md5sum for a message object of type '<SetStreamProfile-request>"
  "d2095f22bd3c757e798fc2eece95a412")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetStreamProfile-request)))
  "Returns md5sum for a message object of type 'SetStreamProfile-request"
  "d2095f22bd3c757e798fc2eece95a412")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetStreamProfile-request>)))
  "Returns full string definition for message of type '<SetStreamProfile-request>"
  (cl:format cl:nil "StreamProfile[] profiles~%~%================================================================================~%MSG: orbbec_camera/StreamProfile~%string stream_name~%int32 width~%int32 height~%int32 fps~%string format~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetStreamProfile-request)))
  "Returns full string definition for message of type 'SetStreamProfile-request"
  (cl:format cl:nil "StreamProfile[] profiles~%~%================================================================================~%MSG: orbbec_camera/StreamProfile~%string stream_name~%int32 width~%int32 height~%int32 fps~%string format~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetStreamProfile-request>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'profiles) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetStreamProfile-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetStreamProfile-request
    (cl:cons ':profiles (profiles msg))
))
;//! \htmlinclude SetStreamProfile-response.msg.html

(cl:defclass <SetStreamProfile-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil)
   (message
    :reader message
    :initarg :message
    :type cl:string
    :initform ""))
)

(cl:defclass SetStreamProfile-response (<SetStreamProfile-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetStreamProfile-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetStreamProfile-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name orbbec_camera-srv:<SetStreamProfile-response> is deprecated: use orbbec_camera-srv:SetStreamProfile-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <SetStreamProfile-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader orbbec_camera-srv:success-val is deprecated.  Use orbbec_camera-srv:success instead.")
  (success m))

(cl:ensure-generic-function 'message-val :lambda-list '(m))
(cl:defmethod message-val ((m <SetStreamProfile-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader orbbec_camera-srv:message-val is deprecated.  Use orbbec_camera-srv:message instead.")
  (message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetStreamProfile-response>) ostream)
  "Serializes a message object of type '<SetStreamProfile-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetStreamProfile-response>) istream)
  "Deserializes a message object of type '<SetStreamProfile-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'message) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'message) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetStreamProfile-response>)))
  "Returns string type for a service object of type '<SetStreamProfile-response>"
  "orbbec_camera/SetStreamProfileResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetStreamProfile-response)))
  "Returns string type for a service object of type 'SetStreamProfile-response"
  "orbbec_camera/SetStreamProfileResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetStreamProfile-response>)))
  "Returns md5sum for a message object of type '<SetStreamProfile-response>"
  "d2095f22bd3c757e798fc2eece95a412")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetStreamProfile-response)))
  "Returns md5sum for a message object of type 'SetStreamProfile-response"
  "d2095f22bd3c757e798fc2eece95a412")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetStreamProfile-response>)))
  "Returns full string definition for message of type '<SetStreamProfile-response>"
  (cl:format cl:nil "bool success~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetStreamProfile-response)))
  "Returns full string definition for message of type 'SetStreamProfile-response"
  (cl:format cl:nil "bool success~%string message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetStreamProfile-response>))
  (cl:+ 0
     1
     4 (cl:length (cl:slot-value msg 'message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetStreamProfile-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetStreamProfile-response
    (cl:cons ':success (success msg))
    (cl:cons ':message (message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetStreamProfile)))
  'SetStreamProfile-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetStreamProfile)))
  'SetStreamProfile-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetStreamProfile)))
  "Returns string type for a service object of type '<SetStreamProfile>"
  "orbbec_camera/SetStreamProfile")