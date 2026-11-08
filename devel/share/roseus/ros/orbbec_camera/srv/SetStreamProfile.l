;; Auto-generated. Do not edit!


(when (boundp 'orbbec_camera::SetStreamProfile)
  (if (not (find-package "ORBBEC_CAMERA"))
    (make-package "ORBBEC_CAMERA"))
  (shadow 'SetStreamProfile (find-package "ORBBEC_CAMERA")))
(unless (find-package "ORBBEC_CAMERA::SETSTREAMPROFILE")
  (make-package "ORBBEC_CAMERA::SETSTREAMPROFILE"))
(unless (find-package "ORBBEC_CAMERA::SETSTREAMPROFILEREQUEST")
  (make-package "ORBBEC_CAMERA::SETSTREAMPROFILEREQUEST"))
(unless (find-package "ORBBEC_CAMERA::SETSTREAMPROFILERESPONSE")
  (make-package "ORBBEC_CAMERA::SETSTREAMPROFILERESPONSE"))

(in-package "ROS")





(defclass orbbec_camera::SetStreamProfileRequest
  :super ros::object
  :slots (_profiles ))

(defmethod orbbec_camera::SetStreamProfileRequest
  (:init
   (&key
    ((:profiles __profiles) ())
    )
   (send-super :init)
   (setq _profiles __profiles)
   self)
  (:profiles
   (&rest __profiles)
   (if (keywordp (car __profiles))
       (send* _profiles __profiles)
     (progn
       (if __profiles (setq _profiles (car __profiles)))
       _profiles)))
  (:serialization-length
   ()
   (+
    ;; orbbec_camera/StreamProfile[] _profiles
    (apply #'+ (send-all _profiles :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; orbbec_camera/StreamProfile[] _profiles
     (write-long (length _profiles) s)
     (dolist (elem _profiles)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; orbbec_camera/StreamProfile[] _profiles
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _profiles (let (r) (dotimes (i n) (push (instance orbbec_camera::StreamProfile :init) r)) r))
     (dolist (elem- _profiles)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(defclass orbbec_camera::SetStreamProfileResponse
  :super ros::object
  :slots (_success _message ))

(defmethod orbbec_camera::SetStreamProfileResponse
  (:init
   (&key
    ((:success __success) nil)
    ((:message __message) "")
    )
   (send-super :init)
   (setq _success __success)
   (setq _message (string __message))
   self)
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:message
   (&optional __message)
   (if __message (setq _message __message)) _message)
  (:serialization-length
   ()
   (+
    ;; bool _success
    1
    ;; string _message
    4 (length _message)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;; string _message
       (write-long (length _message) s) (princ _message s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; string _message
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _message (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass orbbec_camera::SetStreamProfile
  :super ros::object
  :slots ())

(setf (get orbbec_camera::SetStreamProfile :md5sum-) "d2095f22bd3c757e798fc2eece95a412")
(setf (get orbbec_camera::SetStreamProfile :datatype-) "orbbec_camera/SetStreamProfile")
(setf (get orbbec_camera::SetStreamProfile :request) orbbec_camera::SetStreamProfileRequest)
(setf (get orbbec_camera::SetStreamProfile :response) orbbec_camera::SetStreamProfileResponse)

(defmethod orbbec_camera::SetStreamProfileRequest
  (:response () (instance orbbec_camera::SetStreamProfileResponse :init)))

(setf (get orbbec_camera::SetStreamProfileRequest :md5sum-) "d2095f22bd3c757e798fc2eece95a412")
(setf (get orbbec_camera::SetStreamProfileRequest :datatype-) "orbbec_camera/SetStreamProfileRequest")
(setf (get orbbec_camera::SetStreamProfileRequest :definition-)
      "StreamProfile[] profiles

================================================================================
MSG: orbbec_camera/StreamProfile
string stream_name
int32 width
int32 height
int32 fps
string format
---
bool success
string message

")

(setf (get orbbec_camera::SetStreamProfileResponse :md5sum-) "d2095f22bd3c757e798fc2eece95a412")
(setf (get orbbec_camera::SetStreamProfileResponse :datatype-) "orbbec_camera/SetStreamProfileResponse")
(setf (get orbbec_camera::SetStreamProfileResponse :definition-)
      "StreamProfile[] profiles

================================================================================
MSG: orbbec_camera/StreamProfile
string stream_name
int32 width
int32 height
int32 fps
string format
---
bool success
string message

")



(provide :orbbec_camera/SetStreamProfile "d2095f22bd3c757e798fc2eece95a412")


