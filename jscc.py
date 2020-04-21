import os
import glob
import time
from datetime import datetime
import tensorflow as tf
import numpy as np
import configargparse
from tensorflow.keras import layers
import tensorflow_compression as tfc
import data.dataset_cifar10
import data.dataset_imagenet
import data.dataset_kodak

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

DATASETS = {
    "cifar": data.dataset_cifar10,
    "imagenet": data.dataset_imagenet,
    "kodak": data.dataset_kodak,
}


class NBatchLogger(tf.keras.callbacks.Callback):
    """
    A Logger that log average performance per `display` steps.
    """

    def __init__(self, display):
        super(NBatchLogger, self).__init__()
        self.step = 0
        self.display = display
        self.metric_cache = {}
        self._start_time = time.time()

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for k in self.params["metrics"]:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]

        if self.step % self.display == 0:
            cur_time = time.time()
            duration = cur_time - self._start_time
            self._start_time = cur_time
            sec_per_step = duration / self.display

            metrics_log = ""
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += " - %s: %.4f" % (k, val)
                else:
                    metrics_log += " - %s: %.4e" % (k, val)
            print(
                "{} step: {}/{} {} - {:3f} sec/step".format(
                    datetime.now(),
                    self.step,
                    self.params["steps"],
                    metrics_log,
                    sec_per_step,
                )
            )
            self.metric_cache.clear()


class PSNRsVar(tf.keras.metrics.Metric):
    """Calculate the variance of a distribution of PSNRs across batches

    """

    def __init__(self, name="variance", **kwargs):
        super(PSNRsVar, self).__init__(name=name, **kwargs)
        self.count = self.add_weight(name="count", shape=(), initializer="zeros")
        self.mean = self.add_weight(name="mean", shape=(), initializer="zeros")
        self.var = self.add_weight(name="M2", shape=(), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        psnrs = tf.image.psnr(y_true, y_pred, max_val=1.0)
        samples = tf.cast(psnrs, self.dtype)
        batch_count = tf.size(samples)
        batch_count = tf.cast(batch_count, self.dtype)
        batch_mean = tf.math.reduce_mean(samples)
        batch_var = tf.math.reduce_variance(samples)

        # compute new values for variables
        new_count = self.count + batch_count
        new_mean = (self.count * self.mean + batch_count * batch_mean) / (
            self.count + batch_count
        )
        new_var = (
            (self.count * (self.var + tf.square(self.mean - new_mean)))
            + (batch_count * (batch_var + tf.square(batch_mean - new_mean)))
        ) / (self.count + batch_count)

        self.count.assign(new_count)
        self.mean.assign(new_mean)
        self.var.assign(new_var)

    def result(self):
        return self.var

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.count.assign(np.zeros(self.count.shape))
        self.mean.assign(np.zeros(self.mean.shape))
        self.var.assign(np.zeros(self.var.shape))


class TargetPSNRsHistogram(tf.keras.metrics.Metric):
    def __init__(self, name="PSNR target", min_psnr=20, max_psnr=45, step=1, **kwargs):
        super(TargetPSNRsHistogram, self).__init__(name=name, **kwargs)
        self.bins_labels = np.arange(min_psnr, max_psnr + 1, step)
        self.bins = self.add_weight(
            name="bins", shape=self.bins_labels.shape, initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        psnrs = tf.image.psnr(y_true, y_pred, max_val=1.0)
        counts = []
        # count how many images fit in each psnr range
        for b, bin_label in enumerate(self.bins_labels):
            counts.append(tf.math.count_nonzero(tf.greater_equal(psnrs, bin_label)))

        self.bins.assign_add(counts)

    def result(self):
        return self.bins

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.bins.assign(np.zeros(self.bins.shape))


def psnr_metric(x_in, x_out):
    if type(x_in) is list:
        img_in = x_in[0]
    else:
        img_in = x_in
    return tf.image.psnr(img_in, x_out, max_val=1.0)


class Encoder(layers.Layer):
    """Build encoder from specified arch"""

    def __init__(self, conv_depth, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.data_format = "channels_last"
        num_filters = 256
        self.sublayers = [
            tfc.SignalConv2D(
                num_filters,
                (9, 9),
                name="layer_0",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_0"),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_1",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_1"),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_2",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_2"),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_3",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_3"),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                conv_depth,
                (5, 5),
                name="layer_out",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=None,
            ),
        ]

    def call(self, x):
        for sublayer in self.sublayers:
            x = sublayer(x)
        return x


class Decoder(layers.Layer):
    """Build encoder from specified arch"""

    def __init__(self, n_channels, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.data_format = "channels_last"
        num_filters = 256
        self.sublayers = [
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_out",
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_out", inverse=True),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_0",
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_0", inverse=True),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_1",
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_1", inverse=True),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_2",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_2", inverse=True),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                n_channels,
                (9, 9),
                name="layer_3",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.sigmoid,
            ),
        ]

    def call(self, x):
        for sublayer in self.sublayers:
            x = sublayer(x)
        return x


def real_awgn(x, stddev):
    """Implements the real additive white gaussian noise channel.
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # additive white gaussian noise
    awgn = tf.random.normal(tf.shape(x), 0, stddev, dtype=tf.float32)
    y = x + awgn

    return y


def fading(x, stddev, h=None):
    """Implements the fading channel with multiplicative fading and
    additive white gaussian noise.
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # channel gain
    if h is None:
        h = tf.complex(
            tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2)),
            tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2)),
        )

    # additive white gaussian noise
    awgn = tf.complex(
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
    )

    return (h * x + stddev * awgn), h


def phase_invariant_fading(x, stddev, h=None):
    """Implements the fading channel with multiplicative fading and
    additive white gaussian noise. Also assumes that phase shift
    introduced by the fading channel is known at the receiver, making
    the model equivalent to a real slow fading channel.

    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # channel gain
    if h is None:
        n1 = tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2), dtype=tf.float32)
        n2 = tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2), dtype=tf.float32)

        h = tf.sqrt(tf.square(n1) + tf.square(n2))

    # additive white gaussian noise
    awgn = tf.random.normal(tf.shape(x), 0, stddev / np.sqrt(2), dtype=tf.float32)

    return (h * x + awgn), h


class Channel(layers.Layer):
    def __init__(self, channel_type, channel_snr, name="channel", **kwargs):
        super(Channel, self).__init__(name=name, **kwargs)
        self.channel_type = channel_type
        self.channel_snr = channel_snr

    def call(self, inputs):
        (encoded_img, prev_h) = inputs
        inter_shape = tf.shape(encoded_img)
        # reshape array to [-1, dim_z]
        z = layers.Flatten()(encoded_img)
        # convert from snr to std
        print("channel_snr: {}".format(self.channel_snr))
        noise_stddev = np.sqrt(10 ** (-self.channel_snr / 10))

        # Add channel noise
        if self.channel_type == "awgn":
            dim_z = tf.shape(z)[1]
            # normalize latent vector so that the average power is 1
            z_in = tf.sqrt(tf.cast(dim_z, dtype=tf.float32)) * tf.nn.l2_normalize(
                z, axis=1
            )
            z_out = real_awgn(z_in, noise_stddev)
            h = tf.ones_like(z_in)  # h just makes sense on fading channels

        elif self.channel_type == "fading":
            dim_z = tf.shape(z)[1] // 2
            # convert z to complex representation
            z_in = tf.complex(z[:, :dim_z], z[:, dim_z:])
            # normalize the latent vector so that the average power is 1
            z_norm = tf.reduce_sum(
                tf.math.real(z_in * tf.math.conj(z_in)), axis=1, keepdims=True
            )
            z_in = z_in * tf.complex(
                tf.sqrt(tf.cast(dim_z, dtype=tf.float32) / z_norm), 0.0
            )
            z_out, h = fading(z_in, noise_stddev, prev_h)
            # convert back to real
            z_out = tf.concat([tf.math.real(z_out), tf.math.imag(z_out)], 1)

        elif self.channel_type == "fading-real":
            # half of the channels are I component and half Q
            dim_z = tf.shape(z)[1] // 2
            # normalization
            z_in = tf.sqrt(tf.cast(dim_z, dtype=tf.float32)) * tf.nn.l2_normalize(
                z, axis=1
            )
            z_out, h = phase_invariant_fading(z_in, noise_stddev, prev_h)

        else:
            raise Exception("This option shouldn't be an option!")

        # convert signal back to intermediate shape
        z_out = tf.reshape(z_out, inter_shape)
        # compute average power
        avg_power = tf.reduce_mean(tf.math.real(z_in * tf.math.conj(z_in)))
        # add avg_power as layer's metric
        return z_out, avg_power, h


class OutputsCombiner(layers.Layer):
    def __init__(self, name="out_combiner", **kwargs):
        super(OutputsCombiner, self).__init__(name=name, **kwargs)
        self.conv1 = layers.Conv2D(48, 3, 1, padding="same")
        self.prelu1 = layers.PReLU(shared_axes=[1, 2])
        self.conv2 = layers.Conv2D(3, 3, 1, padding="same", activation=tf.nn.sigmoid)

    def call(self, inputs):
        img_prev, residual = inputs

        reconst = tf.concat([img_prev, residual], axis=-1)
        reconst = self.conv1(reconst)
        reconst = self.prelu1(reconst)
        reconst = self.conv2(reconst)

        return reconst


class DeepJSCCF(layers.Layer):
    def __init__(
        self,
        channel_snr,
        conv_depth,
        channel_type,
        feedback_snr,
        refinement_layer,
        layer_id,
        target_analysis=False,
        name="deep_jscc_f",
        **kwargs
    ):
        super(DeepJSCCF, self).__init__(name=name, **kwargs)

        n_channels = 3  # change this if working with BW images
        self.refinement_layer = refinement_layer
        self.feedback_snr = feedback_snr
        self.layer = layer_id
        self.encoder = Encoder(conv_depth)
        self.decoder = Decoder(n_channels, name="decoder_output")
        self.channel = Channel(channel_type, channel_snr, name="channel_output")
        if self.refinement_layer:
            self.image_combiner = OutputsCombiner(name="out_comb")
        self.target_analysis = target_analysis

    def call(self, inputs):
        if self.refinement_layer:
            (
                img,
                prev_img_out_fb,
                prev_chn_out_fb,
                prev_img_out_dec,
                prev_chn_out_dec,
                prev_chn_gain,
            ) = inputs

            img_in = tf.concat([prev_img_out_fb, img], axis=-1)

        else:  # base layer
            # inputs is just the original image
            img_in = img = inputs
            prev_chn_gain = None

        chn_in = self.encoder(img_in)
        chn_out, avg_power, chn_gain = self.channel((chn_in, prev_chn_gain))

        # add feedback noise to chn_output
        if self.feedback_snr is None:  # No feedback noise
            chn_out_fb = chn_out
        else:
            fb_noise_stddev = np.sqrt(10 ** (-self.feedback_snr / 10))
            chn_out_fb = real_awgn(chn_out, fb_noise_stddev)

        if self.refinement_layer:
            # combine chn_output with previous stored chn_outs
            chn_out_exp = tf.concat([chn_out, prev_chn_out_dec], axis=-1)
            residual_img = self.decoder(chn_out_exp)
            # combine residual ith previous stored image reconstruction
            decoded_img = self.image_combiner((prev_img_out_dec, residual_img))

            # feedback estimation
            # Note: the ops below is just computed when this is not the last
            # layer (as this op is not included in the loss function when this
            # is the output), so decoder is just trained with actual chn_outs,
            # and the op below just happens when trainable=False
            chn_out_exp_fb = tf.concat([chn_out_fb, prev_chn_out_fb], axis=-1)
            residual_img_fb = self.decoder(chn_out_exp_fb)
            decoded_img_fb = self.image_combiner([prev_img_out_fb, residual_img_fb])
        else:
            chn_out_exp = chn_out
            decoded_img = self.decoder(chn_out_exp)

            chn_out_exp_fb = chn_out_fb
            decoded_img_fb = self.decoder(chn_out_exp_fb)

        # keep track of some metrics
        self.add_metric(
            tf.image.psnr(img, decoded_img, max_val=1.0),
            aggregation="mean",
            name="psnr{}".format(self.layer),
        )
        self.add_metric(
            tf.image.psnr(img, decoded_img_fb, max_val=1.0),
            aggregation="mean",
            name="psnr_fb{}".format(self.layer),
        )
        self.add_metric(
            tf.reduce_mean(tf.math.square(img - decoded_img)),
            aggregation="mean",
            name="mse{}".format(self.layer),
        )
        self.add_metric(
            avg_power, aggregation="mean", name="avg_pwr{}".format(self.layer)
        )

        return (decoded_img, decoded_img_fb, chn_out_exp, chn_out_exp_fb, chn_gain)

    def change_channel_snr(self, channel_snr):
        self.channel.channel_snr = channel_snr

    def change_feedback_snr(self, feedback_snr):
        self.feedback_snr = feedback_snr


def main(args):
    # get dataset
    x_train, x_val, x_tst = get_dataset(args)

    if args.delete_previous_model and tf.io.gfile.exists(args.model_dir):
        print("Deleting previous model files at {}".format(args.model_dir))
        tf.io.gfile.rmtree(args.model_dir)
        tf.io.gfile.makedirs(args.model_dir)
    else:
        print("Starting new model at {}".format(args.model_dir))
        tf.io.gfile.makedirs(args.model_dir)

    # load model
    prev_layer_out = None
    # add input placeholder to please keras
    img = tf.keras.Input(shape=(None, None, 3))

    if not args.run_eval_once:
        feedback_snr = None if not args.feedback_noise else args.feedback_snr_train
        channel_snr = args.channel_snr_train
    else:
        feedback_snr = None if not args.feedback_noise else args.feedback_snr_eval
        channel_snr = args.channel_snr_eval

    all_models = []
    for layer in range(args.n_layers):
        ckpt_file = os.path.join(args.model_dir, "ckpt_layer{}".format(layer))
        layer_name = "layer{}".format(layer)
        ae_layer = DeepJSCCF(
            channel_snr,
            int(args.conv_depth),
            args.channel,
            feedback_snr,
            layer > 0,  # refinement or base?
            layer,
            args.target_analysis,
            name=layer_name,
        )

        # connect ae_layer to previous model, (if any)
        if layer == 0:  # base layer
            # model returns img and channel outputs
            layer_output = ae_layer(img)
        else:
            # add prev layer outputs as input for cur layer
            (
                prev_img_out_dec,
                prev_img_out_fb,
                prev_chn_out_dec,
                prev_chn_out_fb,
                prev_chn_gain,
            ) = prev_layer_out
            layer_output = ae_layer(
                (
                    img,
                    prev_img_out_fb,
                    prev_chn_out_fb,
                    prev_img_out_dec,
                    prev_chn_out_dec,
                    prev_chn_gain,
                )
            )

        (
            decoded_img,
            _decoded_img_fb,
            _chn_out_exp,
            _chn_out_exp_fb,
            _chn_gain,
        ) = layer_output
        model = tf.keras.Model(inputs=img, outputs=decoded_img)

        model_metrics = [
            tf.keras.metrics.MeanSquaredError(),
            psnr_metric,
            PSNRsVar(name="psnr_var{}".format(layer)),
        ]
        if args.target_analysis:
            model_metrics.append(TargetPSNRsHistogram(name="target{}".format(layer)))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learn_rate),
            loss="mse",
            metrics=model_metrics,
        )

        # check if checkpoint already exists and load it
        if (layer == 0 and args.pretrained_base_layer) or glob.glob(ckpt_file + "*"):
            # trick to restore metrics too (see tensorflow guide on saving and
            # serializing subclassed models)
            model.train_on_batch(x_train)
            if layer == 0 and args.pretrained_base_layer:
                print("Using pre-trained base layer!")
                model.load_weights(
                    os.path.join(
                        args.pretrained_base_layer, "ckpt_layer{}".format(layer)
                    )
                )
            else:
                print("Restoring weights from checkpoint!")
                model.load_weights(ckpt_file)

        print(model.summary())

        # skip training if just running eval or if loading first layer from
        # pretrained ckpt
        if not (args.run_eval_once or (layer == 0 and args.pretrained_base_layer)):
            train_patience = 3 if args.dataset_train != "imagenet" else 2
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=train_patience,
                    monitor="val_psnr_metric",
                    min_delta=10e-3,
                    verbose=1,
                    mode="max",
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.TensorBoard(log_dir=args.eval_dir),
                # just save a single checkpoint with best. If more is wanted,
                # create a new callback
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=ckpt_file,
                    monitor="val_psnr_metric",
                    mode="max",
                    save_best_only=True,
                    verbose=1,
                    save_weights_only=True,
                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ]

            if args.dataset_train == "imagenet":
                callbacks.append(NBatchLogger(100))

            model.fit(
                x_train,
                epochs=args.train_epochs,
                validation_data=x_val,
                callbacks=callbacks,
                verbose=2,
                validation_freq=args.epochs_between_evals,
                validation_steps=(
                    DATASETS[args.dataset_train]._NUM_IMAGES["validation"]
                    // args.batch_size_train
                ),
            )

        # freeze weights of already trained layers
        model.trainable = False
        # define model as prev_model
        prev_layer_out = layer_output
        all_models.append(model)

    print("EVALUATION!!!")
    # normally we just eval the complete model, unless we are doing target_analysis
    models = [model] if not args.target_analysis else all_models
    for eval_model in models:
        out_eval = eval_model.evaluate(x_tst, verbose=2)
        for m, v in zip(eval_model.metrics_names, out_eval):
            met_name = "_".join(["eval", m])
            print("{}={}".format(met_name, v), end=" ")
        print()
        print()


def get_dataset(args):
    data_options = tf.data.Options()
    data_options.experimental_deterministic = False
    data_options.experimental_optimization.apply_default_optimizations = True
    data_options.experimental_optimization.map_parallelization = True
    data_options.experimental_optimization.parallel_batch = True
    data_options.experimental_optimization.autotune_buffers = True

    def prepare_dataset(dataset, mode, parse_record_fn, bs):
        dataset = dataset.with_options(data_options)
        if mode == "train":
            dataset = dataset.shuffle(buffer_size=dataset_obj.SHUFFLE_BUFFER)
        dataset = dataset.map(
            lambda v: parse_record_fn(v, mode, tf.float32),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        return dataset.batch(bs)

    dataset_obj = DATASETS[args.dataset_train]
    parse_record_fn = dataset_obj.parse_record
    if args.dataset_train != "imagenet":
        tr_val_dataset = dataset_obj.get_dataset(True, args.data_dir_train)
        tr_dataset = tr_val_dataset.take(dataset_obj._NUM_IMAGES["train"])
        val_dataset = tr_val_dataset.skip(dataset_obj._NUM_IMAGES["train"])
    else:  # treat imagenet differently, as we usually dont use it for training
        tr_dataset = dataset_obj.get_dataset(True, args.data_dir_train)
        val_dataset = dataset_obj.get_dataset(False, args.data_dir_train)
    # Train
    x_train = prepare_dataset(
        tr_dataset, "train", parse_record_fn, args.batch_size_train
    )
    # Validation
    x_val = prepare_dataset(val_dataset, "val", parse_record_fn, args.batch_size_train)

    # Test
    dataset_obj = DATASETS[args.dataset_eval]
    parse_record_fn = dataset_obj.parse_record
    tst_dataset = dataset_obj.get_dataset(False, args.data_dir_eval)
    x_tst = prepare_dataset(tst_dataset, "test", parse_record_fn, args.batch_size_eval)
    x_tst.repeat(10)  # number of realisations per image on evaluation

    return x_train, x_val, x_tst


if __name__ == "__main__":
    # parse args
    p = configargparse.ArgParser()
    p.add(
        "-c",
        "--my-config",
        required=False,
        is_config_file=True,
        help="config file path",
    )
    p.add(
        "--conv_depth",
        type=float,
        default=16,
        help=(
            "Number of channels of last conv layer, used to define the "
            "compression rate: k/n=c_out/(16*3)"
        ),
        required=True,
    )
    p.add(
        "--n_layers",
        type=int,
        default=3,
        help=("Number of layers/rounds used in the transmission"),
        required=True,
    )
    p.add(
        "--channel",
        type=str,
        default="awgn",
        choices=["awgn", "fading", "fading-real"],
        help="Model of channel used (awgn, fading)",
    )
    p.add(
        "--model_dir",
        type=str,
        default="/tmp/train_logs",
        help=("The location of the model checkpoint files."),
    )
    p.add(
        "--eval_dir",
        type=str,
        default="/tmp/train_logs/eval",
        help=("The location of eval files (tensorboard, etc)."),
    )
    p.add(
        "--delete_previous_model",
        action="store_true",
        default=False,
        help=("If model_dir has checkpoints, delete it before" "starting new run"),
    )
    p.add(
        "--channel_snr_train",
        type=float,
        default=1,
        help="target SNR of channel during training (dB)",
    )
    p.add(
        "--channel_snr_eval",
        type=float,
        default=1,
        help="target SNR of channel during evaluation (dB)",
    )
    p.add(
        "--feedback_noise",
        action="store_true",
        default=False,
        help=("Apply (AWGN) noise to feedback channel"),
    )
    p.add(
        "--feedback_snr_train",
        type=float,
        default=20,
        help=(
            "SNR (dB) of the feedback channel "
            "(only applies when feedback_noise=True)"
        ),
    )
    p.add(
        "--feedback_snr_eval",
        type=float,
        default=20,
        help=(
            "SNR (dB) of the feedback channel (only applies when feedback_noise=True)"
        ),
    )
    p.add(
        "--learn_rate",
        type=float,
        default=0.0001,
        help="Learning rate for Adam optimizer",
    )
    p.add(
        "--run_eval_once",
        action="store_true",
        default=False,
        help="Skip train, run only eval and exit",
    )
    p.add(
        "--train_epochs",
        type=int,
        default=10000,
        help=(
            "The number of epochs used to train (each epoch goes over the whole dataset)"
        ),
    )
    p.add("--batch_size_train", type=int, default=128, help="Batch size for training")
    p.add("--batch_size_eval", type=int, default=128, help="Batch size for evaluation")
    p.add(
        "--epochs_between_evals",
        type=int,
        default=30,
        help=("the number of training epochs to run between evaluations."),
    )
    p.add(
        "--dataset_train",
        type=str,
        default="cifar",
        choices=DATASETS.keys(),
        help=("Choose image dataset. Options: {}".format(DATASETS.keys())),
    )
    p.add(
        "--dataset_eval",
        type=str,
        default="cifar",
        choices=DATASETS.keys(),
        help=("Choose image dataset. Options: {}".format(DATASETS.keys())),
    )
    p.add(
        "--data_dir_train",
        type=str,
        default="/tmp/train_data",
        help="Directory where to store the training data set",
    )
    p.add(
        "--data_dir_eval",
        type=str,
        default="/tmp/train_data",
        help="Directory where to store the eval data set",
    )
    p.add(
        "--pretrained_base_layer",
        type=str,
        help="Use existing checkpoints for base layer",
    )
    p.add(
        "--target_analysis",
        action="store_true",
        default=False,
        help="perform PSNR target analysis",
    )

    args = p.parse_args()

    print("#######################################")
    print("Current execution paramenters:")
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value))
    print("#######################################")
    main(args)
