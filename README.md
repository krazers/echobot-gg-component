### EchoBot Greengrass Component

### MQTT Bridge Merged Parameters
``` json
{
  "mqttTopicMapping": {
        "EchoBotStatusUpdateSubscribe": {
          "topic": "$aws/things/+/shadow/update/delta",
          "source": "IotCore",
          "target": "Pubsub"
        },
        "EchoBotStatusGetSubscribe": {
          "topic": "$aws/things/+/shadow/get/delta",
          "source": "IotCore",
          "target": "Pubsub"
        },
        "EchoBotDetectionsPublish": {
          "topic": "dt/echobot/+/detection",
          "source": "Pubsub",
          "target": "IotCore"
        },
        "EchoBotStatusUpdatePublish": {
          "topic": "$aws/things/+/shadow/update",
          "source": "Pubsub",
          "target": "IotCore"
        },
        "EchoBotStatusGetPublish": {
          "topic": "$aws/things/+/shadow/get",
          "source": "Pubsub",
          "target": "IotCore"
        }
    }
}
```