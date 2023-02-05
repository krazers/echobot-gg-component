###############################################################################
## Global Parameters                                                         ##
###############################################################################
export region=us-east-1
export acct_num=$(aws sts get-caller-identity --query "Account" --output text)
export component_version=0.2.0
corename="echobot_v2"
# CF parameters
export demo_name="echobot"

export artifact_bucket_name=$demo_name-component-artifacts-$acct_num-$region

###############################################################################
## Prereqs                                                                   ##
###############################################################################
echo "###############################################################################"
echo "## Setup prerequisites..."
sudo yum install jq -y

###############################################################################
## Create component that sets up S3 streams in  Stream Manager               ##
## This includes a low and high priority stream. In addition, it monitors    ##
## data transfer state changes from another stream and sends the results to  ##
## IoT Core on topic 'flightdata/status'                                     ##
###############################################################################

echo "###############################################################################"
echo "## Create EchoBot Component for Greengrass..."

# export variables
export component_name=com.demo.echobot.control

# Create artifact for component
mkdir -p ~/GreengrassCore/artifacts/$component_name/$component_version
cp * ~/GreengrassCore/artifacts/$component_name/$component_version -r
(cd ~/GreengrassCore/artifacts/$component_name/$component_version/; zip -m -r $component_name.zip * )

aws s3 mb s3://$artifact_bucket_name

# and copy the artifacts to S3
aws s3 sync ~/GreengrassCore/ s3://$artifact_bucket_name/

# create recipe for component
mkdir -p ~/GreengrassCore/recipes/
touch ~/GreengrassCore/recipes/$component_name-$component_version.json

    uri=s3://$artifact_bucket_name/artifacts/$component_name/$component_version/$component_name.zip
    script="python3 -m pip install awsiotsdk http_parser; python3 -u {artifacts:decompressedPath}/$component_name/echobot.py"
    EchoBotStatusUpdateSubscribe="/things/$corename/shadow/update/delta"
    EchoBotStatusGetSubscribe="/things/$corename/shadow/get/delta"
    EchoBotDetectionsPublish="dt/echobot/$corename/detection"
    EchoBotStatusUpdatePublish="/things/$corename/shadow/update/delta"
    EchoBotStatusGetPublish="/things/$corename/shadow/get"
    json=$(jq --null-input \
    --arg component_name "$component_name" \
    --arg component_version "$component_version" \
    --arg script "$script" \
    --arg uri "$uri" \
    --arg EchoBotStatusUpdateSubscribe "$EchoBotStatusUpdateSubscribe" \
    --arg EchoBotStatusGetSubscribe "$EchoBotStatusGetSubscribe" \
    --arg EchoBotDetectionsPublish "$EchoBotDetectionsPublish" \
    --arg EchoBotStatusUpdatePublish "$EchoBotStatusUpdatePublish" \
    --arg EchoBotStatusGetPublish "$EchoBotStatusGetPublish" \
    '{ "RecipeFormatVersion": "2020-01-25", 
    "ComponentName": $component_name, 
    "ComponentVersion": $component_version, 
    "ComponentDescription": "A component that enables control of EchoBot through shadow updates", 
    "ComponentPublisher": "Azer",
    "ComponentConfiguration": {
        "DefaultConfiguration": {
        "accessControl": {   
                "aws.greengrass.ipc.pubsub": {
                    "<component_name>:pubsub:1": {
                        "policyDescription": "Allows access to publish/subscribe to all topics.",
                        "operations": [
                            "aws.greengrass#PublishToTopic",
                            "aws.greengrass#SubscribeToTopic"
                        ],
                        "resources": [
                            $EchoBotStatusUpdateSubscribe,
                            $EchoBotStatusGetSubscribe,
                            $EchoBotDetectionsPublish,
                            $EchoBotStatusUpdatePublish,
                            $EchoBotStatusGetPublish
                        ]
                    }
                }
            }
        },
        "EchoBotStatusUpdateSubscribe": $EchoBotStatusUpdateSubscribe,
        "EchoBotStatusGetSubscribe": $EchoBotStatusGetSubscribe,
        "EchoBotDetectionsPublish": $EchoBotDetectionsPublish,
        "EchoBotStatusUpdatePublish": $EchoBotStatusUpdatePublish,
        "EchoBotStatusGetPublish": $EchoBotStatusGetPublish
    },
    "Manifests": [ { "Platform": { "os": "linux" }, 
    "Lifecycle": { "RequiresPrivilege": false, "Run": $script }, 
    "Artifacts": [ { "URI": $uri, 
    "Unarchive": "ZIP", "Permission": { "Read": "ALL", "Execute": "NONE" } } ] } ] }')

# Create recipe file and component in Greengrass
echo ${json//<component_name>/$component_name} > ~/GreengrassCore/recipes/$component_name-$component_version.json
aws greengrassv2 create-component-version --inline-recipe fileb://~/GreengrassCore/recipes/$component_name-$component_version.json

echo "###############################################################################"
