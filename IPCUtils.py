# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from json import dumps, loads
from os import getenv
from logging import INFO, DEBUG, StreamHandler, getLogger
import logging, sys

import awsiot.greengrasscoreipc.client as client

from awscrt.io import (
    ClientBootstrap,
    DefaultHostResolver,
    EventLoopGroup,
    SocketDomain,
    SocketOptions,
)
from awsiot.eventstreamrpc import Connection, LifecycleHandler, MessageAmendment
from awsiot.greengrasscoreipc.model import (
    QOS,
    ConfigurationUpdateEvents,
    GetConfigurationRequest,
    PublishToIoTCoreRequest,
    PublishToTopicRequest,
    PublishMessage,
    JsonMessage,
    GetThingShadowRequest,
    UpdateThingShadowRequest,
    SubscribeToTopicRequest,
    SubscriptionResponseMessage
)

# Get a logger
logger = getLogger()
logging.basicConfig(stream=sys.stdout, level=INFO)


TIMEOUT = 15
QOS_TYPE = QOS.AT_LEAST_ONCE

class IPCUtils:
    def connect(self):
        elg = EventLoopGroup()
        resolver = DefaultHostResolver(elg)
        bootstrap = ClientBootstrap(elg, resolver)
        socket_options = SocketOptions()
        socket_options.domain = SocketDomain.Local
        amender = MessageAmendment.create_static_authtoken_amender(getenv("SVCUID"))
        hostname = getenv("AWS_GG_NUCLEUS_DOMAIN_SOCKET_FILEPATH_FOR_COMPONENT")
        connection = Connection(
            host_name=hostname,
            port=8033,
            bootstrap=bootstrap,
            socket_options=socket_options,
            connect_message_amender=amender,
        )
        self.lifecycle_handler = LifecycleHandler()
        connect_future = connection.connect(self.lifecycle_handler)
        connect_future.result(TIMEOUT)
        return connection

    def publish_results_to_cloud(self, topic, PAYLOAD):
        r"""
        Ipc client creates a request and activates the operation to publish messages to the IoT core
        with a qos type over a topic.
        :param PAYLOAD: An dictionary object with inference results.
        """
        try:
            request = PublishToIoTCoreRequest(
                topic_name=topic,
                qos=QOS_TYPE,
                payload=dumps(PAYLOAD).encode(),
            )
            operation = ipc_client.new_publish_to_iot_core()
            operation.activate(request).result(TIMEOUT)
            logger.info("Publishing results to the IoT core...")
            operation.get_response().result(TIMEOUT)
        except Exception as e:
            logger.error(str(e))
            logger.error("Exception occured during publish: {}".format(str(e)))

    def publish_results_to_pubsub_ipc(self, topic, PAYLOAD):
        r"""
        Ipc client creates a request and activates the operation to publish messages to the Greengrass
        IPC Pubsub
        :param PAYLOAD: An dictionary object with inference results.
        """
        try:
            request = PublishToTopicRequest()
            request.topic = topic
            publish_message = PublishMessage()
            publish_message.json_message = JsonMessage()
            publish_message.json_message.message = PAYLOAD
            request.publish_message = publish_message
            operation = ipc_client.new_publish_to_topic()
            logger.info("Publishing results to the Greengrass IPC Pubsub...")
            operation.activate(request)
            future = operation.get_response()
            future.result(TIMEOUT)
        except Exception as e:
            logger.error("Exception occured during publish: {}".format(str(e)))

    def subscribe_to_cloud(self, topic, streamhandler):
        try:
            logger.info("Subscribing to Topic: {}".format(topic))
            request = SubscribeToTopicRequest()
            request.topic_name = topic
            handler = streamhandler
            operation = ipc_client.new_subscribe_to_iot_core(handler) 
            future = operation.activate(request)
            future.result(TIMEOUT)
            logger.info("Subscribed to Topic: {}".format(topic))
        except Exception as e:
            logger.error(
                "Exception occured during subscription: {}".format(str(e))
            )
            exit(1)

    def get_configuration(self):
        r"""
        Ipc client creates a request and activates the operation to get the configuration of
        inference component passed in its recipe.
        :return: A dictionary object of DefaultConfiguration from the recipe.
        """
        try:
            request = GetConfigurationRequest()
            operation = ipc_client.new_get_configuration()
            operation.activate(request).result(TIMEOUT)
            result = operation.get_response().result(TIMEOUT)
            return result.value
        except Exception as e:
            logger.error(
                "Exception occured during fetching the configuration: {}".format(str(e))
            )
            exit(1)
    
    def sample_get_thing_shadow_request(thingName, shadowName):
        try:                    
            # create the GetThingShadow request
            get_thing_shadow_request = GetThingShadowRequest()
            get_thing_shadow_request.thing_name = thingName
            get_thing_shadow_request.shadow_name = shadowName
            
            # retrieve the GetThingShadow response after sending the request to the IPC server
            op = ipc_client.new_get_thing_shadow()
            op.activate(get_thing_shadow_request)
            fut = op.get_response()
            
            result = fut.result(TIMEOUT)
            return result.payload
            
        except Exception as e:
            logger.error(
                "Exception occured during fetching of shadow: {}".format(str(e))
            )
    
    def sample_update_thing_shadow_request(thingName, shadowName, payload):
        try:
            # create the UpdateThingShadow request
            update_thing_shadow_request = UpdateThingShadowRequest()
            update_thing_shadow_request.thing_name = thingName
            update_thing_shadow_request.shadow_name = shadowName
            update_thing_shadow_request.payload = payload
                            
            # retrieve the UpdateThingShadow response after sending the request to the IPC server
            op = ipc_client.new_update_thing_shadow()
            op.activate(update_thing_shadow_request)
            fut = op.get_response()
            
            result = fut.result(TIMEOUT)
            return result.payload
            
        except Exception as e:
            logger.error(
                "Exception occured while updating shadow: {}".format(str(e))
            )


# Get the ipc client
try:
    ipc_client = client.GreengrassCoreIPCClient(IPCUtils().connect())
    logger.info("Created IPC client...")
except Exception as e:
    logger.error(
        "Exception occured during the creation of an IPC client: {}".format(str(e))
    )
    exit(1)