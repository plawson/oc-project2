#!/usr/bin/env bash
aws emr create-cluster \
        --applications Name=Ganglia Name=Spark Name=Zeppelin \
        --ec2-attributes '{"KeyName":"plawson-key-pair-eu-west-1","InstanceProfile":"EMR_EC2_DefaultRole","SubnetId":"subnet-be97b5db","EmrManagedSlaveSecurityGroup":"sg-b83356c3","EmrManagedMasterSecurityGroup":"sg-d73653ac"}' \
        --service-role EMR_DefaultRole \
        --enable-debugging \
        --release-label emr-5.10.0 \
        --log-uri 's3n://aws-logs-641700350459-eu-west-1/elasticmapreduce/' \
        --name 'Cluster OC project 2' \
        --instance-groups '[{"InstanceCount":1,"InstanceGroupType":"MASTER","InstanceType":"m3.xlarge","Name":"Master Instance Group"},{"InstanceCount":4,"InstanceGroupType":"CORE","InstanceType":"m3.xlarge","Name":"Core Instance Group"}]' \
        --configurations file://./emr_config.json \
        --scale-down-behavior TERMINATE_AT_TASK_COMPLETION \
        --region eu-west-1 \
        --bootstrap-action Path=s3://oc-plawson/bootstrap-emr.sh

