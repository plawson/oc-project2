import boto3
from pyspark import SparkContext
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from io import BytesIO
import numpy as np
import sys
import json
import argparse
import matplotlib as mpl
mpl.use('Agg')  # Headless matplotlib usage
import matplotlib.pyplot as plt

sc = SparkContext(appName="animals_identification")


def create_features(config):
    s3 = boto3.resource('s3')
    base_model = VGG16(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
    bucket = s3.Bucket(config['bucket'])

    for obj in bucket.objects.filter(Prefix=config['image_key']):  # Read all jpeg files for the bucket key
        img = image.load_img(BytesIO(obj.get()['Body'].read()), target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        # Save the features to a .txt file
        pos_last_slash = len(obj.key) - obj.key[::-1].find("/")
        filename = obj.key[pos_last_slash:]
        suffix = ".jpg"
        len_suffix = len(suffix)
        pos_last_underscore = len(filename) - filename[::-1].find("_")
        num_file = int(filename[pos_last_underscore:-len_suffix])
        breed_name = filename[:(pos_last_underscore - 1)]
        s3.Object(config['bucket'], config['features_key'] + config['sep']
                  + obj.key[(len(obj.key) - obj.key[::-1].find("/")):] + ".txt") \
            .put(Body=breed_name + ',' + str(num_file) + ',' + json.dumps(features.tolist()[0])[1:][:-1])


def make_labeled_point(features):
    values = [float(x.strip()) for x in features]
    return LabeledPoint(values[0], values[1:])


def s3_object_exists(bucket, file):
    client = boto3.client('s3')
    results = client.list_objects(Bucket=bucket, Prefix=file)
    return 'Contents' in results


def save_graph(results, title, key, config):
    print('save_graph ==============> graph file: {}{}{}{}{}'.format(config['bucket'], config['sep'],
                                                                     config['graph_key'], config['sep'], key))
    x_size = []
    y_percent = []
    for values in results:
        x_size.append(values[0])
        y_percent.append(values[1])

    plt.figure()
    plt.suptitle(title)
    plt.xlabel('Taille du train set')
    plt.ylabel('Pourcentage de réussite')
    plt.plot(x_size, y_percent, 'r')
    plt.xticks(x_size)
    graph = BytesIO()
    plt.savefig(graph, format='jpg')
    graph.seek(0)  # Move the memory stream pointer to the beginning
    s3 = boto3.resource('s3')
    s3.Object(config['bucket'], config['graph_key'] + config['sep'] + key).put(Body=graph)


def do_1vs1(class_one, class_two, size, num_iter, config):
    features_path = config['protocol'] + config['bucket'] + config['sep'] + config['features_key']
    print('do_1vs1 ==============> Setting RDD_ALL')
    rdd_all = sc.textFile(features_path, minPartitions=4).map(lambda line: line.split(',')).persist()
    print('do_1vs1 ==============> Setting RDD_TRAIN_SET')
    rdd_train_set = rdd_all.filter(lambda features: int(features[1]) <= size and (features[0] == class_one
                                                                                  or features[0] == class_two)) \
        .map(lambda features: ['0.0' if features[0] == class_one else '1.0'] + features[2:]) \
        .map(make_labeled_point)

    print('do_1vs1 ==============> Setting RDD_TEST_SET')
    rdd_test_set = rdd_all.filter(lambda features: size < int(features[1]) and (features[0] == class_one
                                                                                              or features[0]
                                                                                              == class_two)) \
        .map(lambda features: ['0.0' if features[0] == class_one else '1.0'] + features[2:]) \
        .map(make_labeled_point)

    # Build the model
    model_dir = class_one + '_' + class_two + '_' + str(size) + '_' + str(num_iter)
    model_s3_file = config['model_key'] + config['sep'] + model_dir
    model = None
    if s3_object_exists(config['bucket'], model_s3_file):
        print('do_1vs1 ==============> Loading SVM Model: {}...'.format(model_s3_file))
        model = SVMModel.load(sc, config['protocol'] + config['bucket'] + config['sep'] + model_s3_file)
    else:
        print('do_1vs1 ==============> Building SVM Model')
        model = SVMWithSGD.train(rdd_train_set, iterations=num_iter)
        print('do_1vs1 ==============> Saving SVM Model: {}...'.format(model_s3_file))
        model.save(sc, config['protocol'] + config['bucket'] + config['sep'] + model_s3_file)

    # Evaluate the model on th test data
    print('do_1vs1 ==============> Evaluating test set')
    labels_and_preds = rdd_test_set.map(lambda p: (p.label, model.predict(p.features)))
    train_err = labels_and_preds.filter(lambda lp: lp[0] != lp[1]).count() / float(rdd_test_set.count())
    # print("Test Error = " + str(train_err))
    success = round(((1 - train_err) * 100), 2)
    print('{},{}'.format(str(size), str(success)))
    return size, success


def do_1vsall(class_all, size, num_iter, config):
    features_path = config['protocol'] + config['bucket'] + config['sep'] + config['features_key']
    print('do_1vsall ==============> Setting RDD_ALL')
    rdd_all = sc.textFile(features_path, minPartitions=4).map(lambda line: line.split(',')).persist()
    print('do_1vsall ==============> Setting RDD_TRAIN_SET')
    rdd_train_set = rdd_all.filter(lambda features: int(features[1]) <= size) \
        .map(lambda features: ['0.0' if features[0] == class_all else '1.0'] + features[2:]) \
        .map(make_labeled_point)

    print('do_1vsall ==============> Setting RDD_TEST_SET')
    rdd_test_set = rdd_all.filter(lambda features: size < int(features[1])) \
        .map(lambda features: ['0.0' if features[0] == class_all else '1.0'] + features[2:]) \
        .map(make_labeled_point)

    # Build the model
    model_dir = class_all + '_' + str(size) + '_' + str(num_iter)
    model_s3_file = config['model_key'] + config['sep'] + model_dir
    model = None
    if s3_object_exists(config['bucket'], model_s3_file):
        print('do_1vsall ==============> Loading SVM Model: {}...'.format(model_s3_file))
        model = SVMModel.load(sc, config['protocol'] + config['bucket'] + config['sep'] + model_s3_file)
    else:
        print('do_1vsall ==============> Building SVM Model')
        model = SVMWithSGD.train(rdd_train_set, iterations=num_iter)
        print('do_1vsall ==============> Saving SVM Model: {}...'.format(model_s3_file))
        model.save(sc, config['protocol'] + config['bucket'] + config['sep'] + model_s3_file)

    # Evaluate the model on th test data
    print('do_1vsall ==============> Evaluating test set')
    labels_and_preds = rdd_test_set.map(lambda p: (p.label, model.predict(p.features)))
    train_err = labels_and_preds.filter(lambda lp: lp[0] != lp[1]).count() / float(rdd_test_set.count())
    # print("Test Error = " + str(train_err))
    success = round(((1 - train_err) * 100), 2)
    print('{},{}'.format(str(size), str(success)))
    return size, success


def main():
    config = {
        "protocol": "s3://",
        "bucket": "oc-plawson",
        "image_key": "distributed_learning/images",
        "sep": "/",
        "features_key": "distributed_learning/features",
        "model_key": "distributed_learning/svm_models",
        "graph_key": "distributed_learning/graph"
    }

    # Define command line parameters
    parser = argparse.ArgumentParser()
    group_class = parser.add_mutually_exclusive_group(required=True)
    group_class.add_argument('--1vs1', help="Classification type, provide classX,classY")
    group_class.add_argument('--1vsAll', help="Classification type, provide classX")
    parser.add_argument('--size', required=True, help="Sizes of the training set separated by commas")
    parser.add_argument('--iter', help="Number of iterations", type=int, default=100)
    parser.add_argument('--graph', help="Create a classification performance graph", action="store_true")

    # Parse and check command line arguments
    print('Parsing command line parameters...')
    args = parser.parse_args()

    # Register input parameters' value
    size_str = args.size
    num_iter = args.iter
    classx_classy = vars(args)['1vs1']
    class_all = vars(args)['1vsAll']

    # Process 1vs1 command line parameters
    class_one = None
    class_two = None
    if None is not classx_classy:
        if not len(classx_classy.split(',')) == 2:
            print("for --1vs1, please provide 2 classes separated by a comma (,) only, you provided: {}".
                  format(classx_classy))
            sys.exit(-1)
        if classx_classy.find(',') == -1 or classx_classy.find(',') == 0 or classx_classy.find(',') == \
                (len(classx_classy) - 1):
            print("for --1vs1, please provide 2 classes separated by a comma (,), you provided: {}".
                  format(classx_classy))
            sys.exit(-1)
        class_one = classx_classy[0:classx_classy.find(',')].strip()
        class_two = classx_classy[(classx_classy.find(',') + 1):].strip()

    if not s3_object_exists(config['bucket'], config['features_key']):
        print('Creating all features...')
        create_features(config)

    results = []
    graph_name = None
    title = None

    if None is not classx_classy:
        for size in [int(x.strip()) for x in size_str.split(',')]:
            print('Execuiting 1VS1, size: {}...'.format(str(size)))
            results.append(do_1vs1(class_one, class_two, size, num_iter, config))
            title = 'Pourcentage de classifications réussies selon la taille du train set en 1 vs 1'
            graph_name = '{}_{}_1vs1_classification_perf.jpg'.format(class_one, class_two)

    if None is not class_all:
        for size in [int(x.strip()) for x in size_str.split(',')]:
            print('Executing 1VSALL, size: {}...'.format(str(size)))
            results.append(do_1vsall(class_all, size, num_iter, config))
            title = 'Pourcentage de classifications réussies selon la taille du train set en 1 vs all'
            graph_name = '{}_1vsall_classification_perf.jpg'.format(class_all)

    if args.graph:
        save_graph(results, title, graph_name, config)


if __name__ == "__main__":
    main()
