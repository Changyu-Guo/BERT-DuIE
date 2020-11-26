# -*- coding: utf - 8 -*-

import os
import configparser
from absl import flags, app

FLAGS = flags.FLAGS


def main(_):

    version_cfg = 'version_' + str(FLAGS.version) + '.cfg'

    config = configparser.ConfigParser()
    config.read(os.path.join('configs', version_cfg))

    if FLAGS.run_data_processor:
        command = 'python3 data_processor.py --version ' + str(FLAGS.version)
        for key, value in config['Data'].items():
            command += ' ' + key + ' ' + value
        os.system(command)

    if FLAGS.run_model:
        common_command = 'python3 model.py --version ' + str(FLAGS.version)
        for key, value in config['Model'].items():
            common_command += ' ' + key + ' ' + value

        if FLAGS.run_train:
            common_command += ' --run_train True'

        if FLAGS.run_evaluate:
            common_command += ' --run_evaluate True'

        if FLAGS.run_inference:
            common_command += ' -- run_inference True'

        os.system(common_command)


if __name__ == '__main__':

    flags.DEFINE_integer(
        name='version',
        default=None,
        help='Version.'
    )
    flags.mark_flag_as_required('version')

    flags.DEFINE_boolean(
        name='run_data_processor',
        default=False,
        help='Whether run data processor to process data.'
    )

    flags.DEFINE_boolean(
        name='run_model',
        default=True,
        help='Whether run model file to train or evaluate or inference model.'
    )

    flags.DEFINE_boolean(
        name='run_train',
        default=True,
        help='Whether run train function in  model file.'
    )

    flags.DEFINE_boolean(
        name='run_evaluate',
        default=False,
        help='Whether run evaluate function in model file.'
    )

    flags.DEFINE_boolean(
        name='run_inference',
        default=False,
        help='Whether run inference function in model file.'
    )

    app.run(main)
