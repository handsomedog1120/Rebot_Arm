;; Auto-generated. Do not edit!


(when (boundp 'orbbec_camera::StreamProfile)
  (if (not (find-package "ORBBEC_CAMERA"))
    (make-package "ORBBEC_CAMERA"))
  (shadow 'StreamProfile (find-package "ORBBEC_CAMERA")))
(unless (find-package "ORBBEC_CAMERA::STREAMPROFILE")
  (make-package "ORBBEC_CAMERA::STREAMPROFILE"))

(in-package "ROS")
;;//! \htmlinclude StreamProfile.msg.html


(defclass orbbec_camera::StreamProfile
  :super ros::object
  :slots (_stream_name _width _height _fps _format ))

(defmethod orbbec_camera::StreamProfile
  (:init
   (&key
    ((:stream_name __stream_name) "")
    ((:width __width) 0)
    ((:height __height) 0)
    ((:fps __fps) 0)
    ((:format __format) "")
    )
   (send-super :init)
   (setq _stream_name (string __stream_name))
   (setq _width (round __width))
   (setq _height (round __height))
   (setq _fps (round __fps))
   (setq _format (string __format))
   self)
  (:stream_name
   (&optional __stream_name)
   (if __stream_name (setq _stream_name __stream_name)) _stream_name)
  (:width
   (&optional __width)
   (if __width (setq _width __width)) _width)
  (:height
   (&optional __height)
   (if __height (setq _height __height)) _height)
  (:fps
   (&optional __fps)
   (if __fps (setq _fps __fps)) _fps)
  (:format
   (&optional __format)
   (if __format (setq _format __format)) _format)
  (:serialization-length
   ()
   (+
    ;; string _stream_name
    4 (length _stream_name)
    ;; int32 _width
    4
    ;; int32 _height
    4
    ;; int32 _fps
    4
    ;; string _format
    4 (length _format)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _stream_name
       (write-long (length _stream_name) s) (princ _stream_name s)
     ;; int32 _width
       (write-long _width s)
     ;; int32 _height
       (write-long _height s)
     ;; int32 _fps
       (write-long _fps s)
     ;; string _format
       (write-long (length _format) s) (princ _format s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _stream_name
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _stream_name (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; int32 _width
     (setq _width (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _height
     (setq _height (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _fps
     (setq _fps (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; string _format
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _format (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(setf (get orbbec_camera::StreamProfile :md5sum-) "eab518462680efdcb8df71e9c4ed57d9")
(setf (get orbbec_camera::StreamProfile :datatype-) "orbbec_camera/StreamProfile")
(setf (get orbbec_camera::StreamProfile :definition-)
      "string stream_name
int32 width
int32 height
int32 fps
string format

")



(provide :orbbec_camera/StreamProfile "eab518462680efdcb8df71e9c4ed57d9")


