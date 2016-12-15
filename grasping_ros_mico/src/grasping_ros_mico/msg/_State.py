"""autogenerated by genpy from grasping_ros_mico/State.msg. Do not edit."""
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import geometry_msgs.msg
import std_msgs.msg

class State(genpy.Message):
  _md5sum = "1b5954efa74e0ab337859b68d907407b"
  _type = "grasping_ros_mico/State"
  _has_header = False #flag to mark the presence of a Header object
  _full_text = """geometry_msgs/PoseStamped gripper_pose
geometry_msgs/PoseStamped object_pose
int32 observation

================================================================================
MSG: geometry_msgs/PoseStamped
# A Pose with reference coordinate frame and timestamp
Header header
Pose pose

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
# 0: no frame
# 1: global frame
string frame_id

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of postion and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

"""
  __slots__ = ['gripper_pose','object_pose','observation']
  _slot_types = ['geometry_msgs/PoseStamped','geometry_msgs/PoseStamped','int32']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       gripper_pose,object_pose,observation

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(State, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.gripper_pose is None:
        self.gripper_pose = geometry_msgs.msg.PoseStamped()
      if self.object_pose is None:
        self.object_pose = geometry_msgs.msg.PoseStamped()
      if self.observation is None:
        self.observation = 0
    else:
      self.gripper_pose = geometry_msgs.msg.PoseStamped()
      self.object_pose = geometry_msgs.msg.PoseStamped()
      self.observation = 0

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      _x = self
      buff.write(_struct_3I.pack(_x.gripper_pose.header.seq, _x.gripper_pose.header.stamp.secs, _x.gripper_pose.header.stamp.nsecs))
      _x = self.gripper_pose.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      if python3:
        buff.write(struct.pack('<I%sB'%length, length, *_x))
      else:
        buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_struct_7d3I.pack(_x.gripper_pose.pose.position.x, _x.gripper_pose.pose.position.y, _x.gripper_pose.pose.position.z, _x.gripper_pose.pose.orientation.x, _x.gripper_pose.pose.orientation.y, _x.gripper_pose.pose.orientation.z, _x.gripper_pose.pose.orientation.w, _x.object_pose.header.seq, _x.object_pose.header.stamp.secs, _x.object_pose.header.stamp.nsecs))
      _x = self.object_pose.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      if python3:
        buff.write(struct.pack('<I%sB'%length, length, *_x))
      else:
        buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_struct_7di.pack(_x.object_pose.pose.position.x, _x.object_pose.pose.position.y, _x.object_pose.pose.position.z, _x.object_pose.pose.orientation.x, _x.object_pose.pose.orientation.y, _x.object_pose.pose.orientation.z, _x.object_pose.pose.orientation.w, _x.observation))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(_x))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(_x))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    try:
      if self.gripper_pose is None:
        self.gripper_pose = geometry_msgs.msg.PoseStamped()
      if self.object_pose is None:
        self.object_pose = geometry_msgs.msg.PoseStamped()
      end = 0
      _x = self
      start = end
      end += 12
      (_x.gripper_pose.header.seq, _x.gripper_pose.header.stamp.secs, _x.gripper_pose.header.stamp.nsecs,) = _struct_3I.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.gripper_pose.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.gripper_pose.header.frame_id = str[start:end]
      _x = self
      start = end
      end += 68
      (_x.gripper_pose.pose.position.x, _x.gripper_pose.pose.position.y, _x.gripper_pose.pose.position.z, _x.gripper_pose.pose.orientation.x, _x.gripper_pose.pose.orientation.y, _x.gripper_pose.pose.orientation.z, _x.gripper_pose.pose.orientation.w, _x.object_pose.header.seq, _x.object_pose.header.stamp.secs, _x.object_pose.header.stamp.nsecs,) = _struct_7d3I.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.object_pose.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.object_pose.header.frame_id = str[start:end]
      _x = self
      start = end
      end += 60
      (_x.object_pose.pose.position.x, _x.object_pose.pose.position.y, _x.object_pose.pose.position.z, _x.object_pose.pose.orientation.x, _x.object_pose.pose.orientation.y, _x.object_pose.pose.orientation.z, _x.object_pose.pose.orientation.w, _x.observation,) = _struct_7di.unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      _x = self
      buff.write(_struct_3I.pack(_x.gripper_pose.header.seq, _x.gripper_pose.header.stamp.secs, _x.gripper_pose.header.stamp.nsecs))
      _x = self.gripper_pose.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      if python3:
        buff.write(struct.pack('<I%sB'%length, length, *_x))
      else:
        buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_struct_7d3I.pack(_x.gripper_pose.pose.position.x, _x.gripper_pose.pose.position.y, _x.gripper_pose.pose.position.z, _x.gripper_pose.pose.orientation.x, _x.gripper_pose.pose.orientation.y, _x.gripper_pose.pose.orientation.z, _x.gripper_pose.pose.orientation.w, _x.object_pose.header.seq, _x.object_pose.header.stamp.secs, _x.object_pose.header.stamp.nsecs))
      _x = self.object_pose.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      if python3:
        buff.write(struct.pack('<I%sB'%length, length, *_x))
      else:
        buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_struct_7di.pack(_x.object_pose.pose.position.x, _x.object_pose.pose.position.y, _x.object_pose.pose.position.z, _x.object_pose.pose.orientation.x, _x.object_pose.pose.orientation.y, _x.object_pose.pose.orientation.z, _x.object_pose.pose.orientation.w, _x.observation))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(_x))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(_x))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    try:
      if self.gripper_pose is None:
        self.gripper_pose = geometry_msgs.msg.PoseStamped()
      if self.object_pose is None:
        self.object_pose = geometry_msgs.msg.PoseStamped()
      end = 0
      _x = self
      start = end
      end += 12
      (_x.gripper_pose.header.seq, _x.gripper_pose.header.stamp.secs, _x.gripper_pose.header.stamp.nsecs,) = _struct_3I.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.gripper_pose.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.gripper_pose.header.frame_id = str[start:end]
      _x = self
      start = end
      end += 68
      (_x.gripper_pose.pose.position.x, _x.gripper_pose.pose.position.y, _x.gripper_pose.pose.position.z, _x.gripper_pose.pose.orientation.x, _x.gripper_pose.pose.orientation.y, _x.gripper_pose.pose.orientation.z, _x.gripper_pose.pose.orientation.w, _x.object_pose.header.seq, _x.object_pose.header.stamp.secs, _x.object_pose.header.stamp.nsecs,) = _struct_7d3I.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.object_pose.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.object_pose.header.frame_id = str[start:end]
      _x = self
      start = end
      end += 60
      (_x.object_pose.pose.position.x, _x.object_pose.pose.position.y, _x.object_pose.pose.position.z, _x.object_pose.pose.orientation.x, _x.object_pose.pose.orientation.y, _x.object_pose.pose.orientation.z, _x.object_pose.pose.orientation.w, _x.observation,) = _struct_7di.unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill

_struct_I = genpy.struct_I
_struct_7di = struct.Struct("<7di")
_struct_3I = struct.Struct("<3I")
_struct_7d3I = struct.Struct("<7d3I")
