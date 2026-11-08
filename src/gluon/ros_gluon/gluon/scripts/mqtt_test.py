import paho.mqtt.client as mqtt

client = mqtt.Client()
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker!")
    else:
        print("Connection failed. Return code =", rc)

def on_message(client, userdata, msg):
    print("Received message: ", str(msg.payload.decode("utf-8")))

client.on_connect = on_connect
client.on_message = on_message

broker = "192.168.0.102"  # 替换为您的MQTT代理服务器地址
port = 1883  # 默认MQTT端口号

client.connect(broker, port)

client.loop_start()

topic = "leapmotioninfo"  # 替换为您希望订阅的主题
client.subscribe(topic)
message = "Hello, MQTT!"
client.publish(topic, message)
client.loop_stop()
client.disconnect()
