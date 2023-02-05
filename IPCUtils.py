# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from json import dumps
from os import getenv
import config_utils

from awsiot.greengrasscoreipc.clientv2 import GreengrassCoreIPCClientV2
from awsiot.greengrasscoreipc.model import (
    PublishMessage,
    BinaryMessage,
    SubscriptionResponseMessage,
    UnauthorizedError,
    GetConfigurationRequest
)

ipc_client = None

def publish_results_to_pubsub_ipc(topic, message):
    return ipc_client.publish_to_topic(topic=topic, publish_message=message)

def subscribe_to_topic(topic,on_stream_event,on_stream_error,on_stream_closed):
    _, operation = ipc_client.subscribe_to_topic(topic=topic, on_stream_event=on_stream_event,
                                                     on_stream_error=on_stream_error, on_stream_closed=on_stream_closed)
    config_utils.logger.info('Successfully subscribed to topic: ' + topic)

def get_configuration(self):
        r"""
        Ipc client creates a request and activates the operation to get the configuration of
        inference component passed in its recipe.

        :return: A dictionary object of DefaultConfiguration from the recipe.
        """
        try:
            request = GetConfigurationRequest()
            operation = ipc_client.new_get_configuration()
            operation.activate(request).result(config_utils.TIMEOUT)
            result = operation.get_response().result(config_utils.TIMEOUT)
            return result.value
        except Exception as e:
            config_utils.logger.error(
                "Exception occured during fetching the configuration: {}".format(e)
            )
            exit(1)

# Get the ipc client
try:
    ipc_client = GreengrassCoreIPCClientV2()
    config_utils.logger.info("Created IPC client...")
except Exception as e:
    config_utils.logger.error(
        "Exception occured during the creation of an IPC client: {}".format(e)
    )
    exit(1)
