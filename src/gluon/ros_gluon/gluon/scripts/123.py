if __name__ == '__main__':
    try:
        print("Start control")
        publisher = ros_publisher()
        # client = connect_mqtt()
        # sub_send2ros(client, topic, publisher)
        # client.loop_forever()
        # t1 = threading.Thread(target=client.loop_start, args=())
        # t1.start()

    except rospy.ROSInterruptException:
        print('over')
        pass
