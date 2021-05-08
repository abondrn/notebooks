import os
import re
import argparse
import random
import warnings
import shutil
import glob
import traceback
import logging
import itertools
import time

import sh
from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sklearn
import bids
import nilearn.image
import nilearn.input_data
import nilearn.datasets
import nilearn.plotting


SOURCE = 's3://openneuro/ds000030/ds000030_R1.0.5/uncompressed'

s3_sync = sh.aws.s3.sync.bake('--no-sign-request')
s3_ls = sh.aws.s3.ls.bake('--no-sign-request')


def run(cmd, trace):
    try:
        proc = cmd(_out=lambda line, stdin: print(line), _bg=True)
        if trace:
            print(proc.ran.replace('"', '\\"'))
        proc.wait()
    except KeyboardInterrupt:
        logging.info('Ending process...')
        proc.terminate()
        raise


def list_data(path='', files=None, trace=False):
    if path != '' and not path.endswith('/'):
        path += '/'
    cmd = s3_ls.bake(
        os.path.join(SOURCE, path),
    )
    if files:
        cmd = cmd.bake(exclude='*')
        for file in files:
            cmd = cmd.bake(include=file)

    run(cmd, trace)


def include_data(folder, path, files=None, override=False, trace=False):
    '''Downloads files from the dataset'''
    cmd = s3_sync.bake(
        os.path.join(SOURCE, path),
        os.path.join(folder, path))
    if files:
        if os.path.exists(path) and not override:
            print(path, 'already downloaded')
            if not override:
                missing = []
                for file in files:
                    if os.path.exists(os.path.join(folder, path, file)):
                        print(file, 'already downloaded')
                    else:
                        missing.append(include=file)
                files = missing
        cmd = cmd.bake(exclude='*')
        for file in files:
            cmd = cmd.bake(include=file)

    run(cmd, trace)


def clean_data(folder, path, files=None, override=False, trace=False):
    '''Cleans files that are no longer needed from the dataset'''
    matches = []
    if files:
        for file in files:
            matches.extend(glob.glob(os.path.join(folder, path, file.replace('*', '**'))))
    else:
        matches.append(os.path.join(folder, path))
    for match in matches:
        print('Removing', match)
        if os.path.exists(match):
            if os.path.isfile(match):
                os.remove(match)
            else:
                shutil.rmtree(match)


def init(folder):
    include_data(folder, '', ['participants.tsv', 'dataset_description.json'])
    include_data(folder, 'phenotype')


def subject_data_action(action, subject, folder,
                         modalities=['anat', 'func'], func_patterns=None,
                         unprocessed=False,
                         derivatives=[], **kwargs):
    '''Acts upon (i.e., downloads or cleans) the data associated with a subject'''
    roots = []
    op = {
        'download': include_data,
        'clean': clean_data,
    }[action]
    for deriv in derivatives:
        if deriv == 'unprocessed':
            roots.append(subject)
        else:
            roots.append(os.path.join('derivatives', deriv, subject))
    for root in roots:
        for mod in modalities:
            if mod == 'func' and func_patterns:
                op(folder, os.path.join(root, mod), files=func_patterns, **kwargs)
            else:
                op(folder, os.path.join(root, mod), **kwargs)


def include_subjects_data(subjects, folder, tasks=['all'], foreach=None, skip=None,
                          keep=True, progress=False, local=False,
                          *args, **kwargs):
    interrupted = False
    if progress:
        subjects = tqdm(subjects, total=len(subjects))
    func_patterns = []
    for task in tasks:
        if task == 'all':
            func_patterns.append('*task-*_bold*') # .nii.gz
        else:
            func_patterns.append(f'*task-{task}_bold*')
    for sub in subjects:
        try:
            if not kwargs.get('override') and skip and skip(sub, folder, tasks):
                print('Skipping', sub)
            else:
                if not local:
                    print('Fetching data for', sub)
                    subject_data_action('download', sub, folder, func_patterns=func_patterns, *args, **kwargs)
                if foreach:
                    print('Processing', sub)
                    foreach(sub, folder, tasks)
        except:
            traceback.print_exc()
            logging.warning('Aborting...')
            interrupted = True
        finally:
            if not keep:
                print('Cleaning up raw data for', sub)
                subject_data_action('clean', sub, folder, func_patterns=func_patterns, *args, **kwargs)
            if interrupted:
                return


def plot_rois(map, labels, legend_labels=None):
    if legend_labels is None:
        legend_labels = labels
    fig, ax1 = plt.subplots()
    RGB = 'R G B'.split()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', labels[RGB].values, len(labels)+1)
    nilearn.plotting.plot_roi(map,
                              title='Cortical parcelation into %d regions' % len(labels),
                              cmap=cmap,
                              draw_cross=False, cut_coords=(-1,8,9),
                              axes=ax1)
    patches = [matplotlib.patches.Patch(color=list(r[RGB]), label=r.Name) for idx, r in legend_labels.iterrows()]
    divider = make_axes_locatable(ax1)
    ax2 = divider.append_axes('right',size='0%')
    ax2.legend(handles=patches, title='ROIs', fontsize='small', loc='center left')
    ax2.axis('off')


def load_yeo7_atlas():
    '''
    The blurred, 7-network estimate from Yeo et al. 2011.
    Uses the assigned colors, and ROI names inferred from Figure 12 of the Yeo paper.
    '''
    yeo_atlas = nilearn.datasets.fetch_atlas_yeo_2011('rois')
    yeo_7 = nilearn.image.load_img(yeo_atlas.thick_7)

    RGB = 'R G B'.split()
    labels7 = pd.read_csv(yeo_atlas.colors_7, skiprows=1,
                        sep='\s+', index_col=0, names='Label R G B Name'.split())
    labels7.loc[:,'Name'] = ['Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Frontoparietal', 'Default']
    labels7[RGB] = labels7[RGB] / 255

    return yeo_7, labels7


def load_yeo17_atlas():
    yeo_atlas = nilearn.datasets.fetch_atlas_yeo_2011('rois')
    yeo_17 = nilearn.image.load_img(yeo_atlas.thick_17)

    RGB = 'R G B'.split()
    labels17 = pd.read_csv(yeo_atlas.colors_17, skiprows=1,
                        sep='\s+', index_col=0, names='Label R G B Name'.split())
    labels17['Name'] = ['Central Visual', 'Peripheral Visual', 'Somatomotor A', 'Somatomotor B', 'Dorsal Attention A', 'Dorsal Attention B', 'Salience / Ventral Attention A', 'Salience / Ventral Attention B',
                        'Limbic B', 'Limbic A', 'Control C', 'Control A', 'Control B', 'Temporal Parietal', 'Default C', 'Default A', 'Default B']
    labels17['Parent'] = ['Visual', 'Visual', 'Somatomotor', 'Somatomotor', 'Dorsal Attention', 'Dorsal Attention', 'Ventral Attention', 'Ventral Attention',
                        'Limbic', 'Limbic', 'Frontoparietal', 'Frontoparietal', 'Frontoparietal', 'Ventral Attention', 'Default', 'Default', 'Default']
    labels17[RGB] = labels17[RGB] / 255

    return yeo_17, labels17


def load_schaefer100_atlas(labels17=None):
    if labels17 is None:
        _, labels17 = load_yeo17_atlas()
    schaefer_atlas = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=17, data_dir='rois')
    schaefer_100 = nilearn.image.load_img(schaefer_atlas.maps)
    components = pd.read_csv('https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Yeo2011_fcMRI_clustering/1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels/Yeo2011_17networks_N1000.split_components.glossary.csv',
                            skiprows=2, names=['Component Label', 'Network Name', 'Component Name'])
    components['Component Name'] = components['Component Name'].str.title()
    components['Network Name'] = components['Network Name'].str.title().str.rstrip()
    components = components.merge(labels17.drop(columns=['Label']), how='left', left_on='Network Name', right_on='Name')

    def expand_label(label):
        s = label.decode()
        return (s, s.split('_', maxsplit=2)[1], s.rsplit('_', maxsplit=1)[0])
    labels100 = pd.DataFrame([expand_label(l) for l in schaefer_atlas.labels],
        columns=['Label', 'Side', 'Component Label']
        ).merge(components, how='left', on='Component Label')

    return schaefer_100, labels100


def extract_confounds(file):
    return pd.read_csv(file, sep='\t')


def filter_files(results, modality, type, suffix=None, task=None):
    return [r for r in results if all(part in r for part in (modality, type, suffix, task) if part)]


def preprocess(subject, layout, maskers, task=None):
    drop_rows = 1
    match = re.fullmatch('sub-(\d+)', subject)
    assert match, 'Invalid subject'
    sid = match.group(1)
    
    files = list(layout.get(subject=sid, return_type='file'))
    # Load preprocessed T1-weighted fMRI
    func_files = filter_files(files,
                            modality='func',
                            type='preproc',
                            task=task,
                            suffix='T1w')
    # Load confound transformations
    confound_files = filter_files(files,
                                modality='func',
                                task=task,
                                type='confounds')
    if len(func_files) != 1:
        logging.warning('Duplicate or no functional files found '+str(func_files))
        raise FileNotFoundError
    if len(confound_files) != 1:
        logging.warning('Duplicate or no confounder files found'+str(confound_files))
        raise FileNotFoundError

    func_img = nilearn.image.load_img(func_files[0])
    func_img = func_img.slicer[:,:,:,drop_rows:]
    
    confounds = extract_confounds(confound_files[0]).values[drop_rows:,:]
    
    out = []
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', module='nilearn|scipy')
        for m in maskers:
            out.append(m.fit_transform(func_img,[confounds]))
    return out


class Processor:

    def __init__(self, derivatives):
        yeo_7, _ = load_yeo7_atlas()
        yeo_17, labels17 = load_yeo17_atlas()
        schaefer_100, _ = load_schaefer100_atlas(labels17)
        common = dict(
            strategy='mean',
            memory='nilearn_cache',
            detrend=True, # removes linear trends from the data
            low_pass = 0.08, # cleans the signal of fluctuations above a certain frequency band (used in many RS analyses incl. Yeo)
            high_pass = 0.009,
            t_r=2, # repetition time: time between samples (sourced from the CNP study)
            standardize=False,
        )
        self.derivatives = derivatives
        self.maskers = {
            7: nilearn.input_data.NiftiLabelsMasker(
                labels_img=yeo_7, **common),
            17: nilearn.input_data.NiftiLabelsMasker(
                labels_img=yeo_17, **common),
            100: nilearn.input_data.NiftiLabelsMasker(
                labels_img=schaefer_100, **common),
        }

    def process_callback(self, s, folder, tasks):
        layout = get_layout(folder, self.derivatives)
        maskeys = list(self.maskers.keys())
        for task in tasks:
            try:
                masked = preprocess(s, layout, [self.maskers[k] for k in maskeys], task)
            except FileNotFoundError:
                continue
            where = os.path.join(folder, s)
            if not os.path.exists(where):
                os.makedirs(where)
            for masked, parcels in zip(masked, maskeys):
                np.save(os.path.join(where, f'{task}_{parcels}'), masked)

    def skip_callback(self, s, folder, tasks):
        for task in tasks:
            for parcels in [7, 17, 100]:
                if not os.path.exists(os.path.join(folder, s, f'{task}_{parcels}.npy')):
                    return False
        return True


def load_data(subject, folder):
    dct = {}
    for fn in glob.glob(os.path.join(folder, subject, '*_*.npy')):
        name = os.path.splitext(os.path.basename(fn))[0]
        dct[name] = np.load(fn)
    return dct


def get_layout(folder, derivatives):
    layout = bids.layout.BIDSLayout(folder, index_metadata=False)
    for deriv in derivatives:
        if deriv != 'unprocessed':
            layout.add_derivatives(os.path.join(folder, 'derivatives', deriv), index_metadata=False)
    return layout


def downloaded_subjects(layout):
    return [f'sub-{sid}' for sid in layout.get_subjects()]


def snake_to_camel(string):
    return ''.join(word.capitalize() or '_' for word in string.split('_'))


class DummyTransformer(sklearn.base.TransformerMixin):

    def fit(self, X, y=None):
        '''Ignore the data passed in.'''
        return self

    def transform(self, X, y=None):
        '''Return the input unaltered.'''
        return X
    
    def inverse_transform(self, X):
        return X


def transformer(name=None):
    name = name or snake_to_camel(callback.__name__)
    def wrapper(func):
        return type(name, (DummyTransformer,), {'transform': staticmethod(func)})()
    return wrapper


def estimator_to_transformer(estimator, **kwargs):
    return type(estimator.__name__+'Transformer', (estimator,), {'transform': estimator.predict})(**kwargs)


def add_transformer_hook(transformer, hook, suffix='Hook', **kwargs):
    return type(transformer.__name__+suffix, (transformer,), {'transform':
        lambda self, *args, **kwargs: hook(transformer.transform(self, *args, **kwargs))})(**kwargs)


def num_params(model):
    model = model or self.current[-1]
    k = 0
    for attr, val in model.__dict__.items():
        if isinstance(val, np.ndarray) and val.dtype == float:
            k += val.size
    return k


def accuracy(model, X, y):
    return sklearn.metrics.accuracy_score(y, model.predict(X))


def aic(model, X, y, k=None):
    '''Modified AIC'''
    k = k or num_params(model)
    LL = sklearn.metrics.log_loss(y, model.predict_proba(X), labels=model.classes_)
    return LL - k/len(y)


# via https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html?highlight=roc%20auc
# TODO: class labels
def roc(model, X, y, plot=False, multiclass=False, name='', figsize=(8,8)):
    auc = dict()
    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)

    for i, cls in enumerate(model.classes_):
        fpr, tpr, thres = sklearn.metrics.roc_curve(
            y == cls,
            model.predict_proba(X)[:, i])
        auc[cls] = sklearn.metrics.auc(fpr, tpr)
        if plot:
            ax.plot(fpr, tpr,
                label='%s (AUC = %0.2f)' % (cls, auc[cls]), lw=1)
            interp_tpr = scipy.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

    if plot:
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(list(auc.values()))
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               xlabel='FPR', ylabel='TPR',
               title=name+' Receiver operating characteristic')
        ax.legend(loc="lower right")
        plt.show()
    return auc


# TODO reduced to _ features
class Pipeline(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):

    def __init__(self, name, estimator, transformers):
        '''Initialize self. See help(type(self)) for accurate signature.'''
        self.name = name
        self.estimator = estimator
        self.transformers = transformers
    
    def transform(self, X, y=None, **kwargs):
        for tname, transformer in self.transformers:
            X = transformer.transform(X)
        return X

    def fit(self, X, y, **kwargs):
        for tname, transformer in self.transformers:
            X = transformer.fit_transform(X, kwargs[tname] if tname in kwargs else y)
        self.estimator.fit(X, y)
        return self

    def predict(self, X, **kwargs):
        return self.estimator.predict(self.transform(X, **kwargs))

    def predict_proba(self, X, **kwargs):
        return self.estimator.predict_proba(self.transform(X, **kwargs))

    def inverse_transform(self, X, **kwargs):
        for tname, transformer in reversed(self.transformers):
            if tname in kwargs:
                X = transformer.inverse_transform(X, kwargs[tname])
            else:
                X = transformer.inverse_transform(X)
        return X

    def decision_function(self, X, **kwargs):
        return self.estimator.decision_function(self.transform(X, **kwargs))

    def __getattr__(self, attr):
        return getattr(self.estimator, attr)


class PipelineGrid:

    def __init__(self, estimators, features=None, transformers=[], metrics=[], progress=False):
        self.features = features
        self.transformers = []
        for ts in transformers:
            if type(ts) is tuple:
                self.transformers.append([ts])
            elif type(ts) is dict:
                if len(ts) == 1:
                    name, t = next(iter(ts.items()))
                    self.transformers.append([
                        (None, DummyTransformer()),
                        (name, t),
                    ])
                else:
                    self.transformers.append(list(ts.items()))
            else:
                self.transformers.append([(None, ts)])
        self.estimators = estimators
        self.metrics = metrics
        self.progress = progress
    
    def __iter__(self):
        pipelines = itertools.product(*
            [list(self.features.items())]+self.transformers+[list(self.estimators.items())])
        if self.progress:
            num = len(self.features) * len(self.estimators)
            for ts in self.transformers:
                num *= len(ts)
            pipelines = tqdm(pipelines, total=num)
        self.pipelines = iter(pipelines)
        return self

    def __next__(self):
        try:
            current = next(self.pipelines)
            return current[0][1], Pipeline(
                name=tuple([n for n, _ in current]),
                transformers=current[1:-1],
                estimator=current[-1][1])
        except StopIteration:
            raise

    def get_model(self, name):
        return self.features[name[0]], Pipeline(
            name=name,
            transformers=[(n, dict(self.transformers[i])[n]) for i, n in enumerate(name[1:-1])],
            estimator=self.estimators[name[-1]])

    def score_all(self, fname, X_train, y_train, X_test, y_test, **kwargs):
        pipelines = itertools.product(*
            self.transformers+[list(self.estimators.items())])
        if self.progress:
            num = len(self.estimators)
            for ts in self.transformers:
                num *= len(ts)
            pipelines = tqdm(pipelines, total=num)
        prev = [('placeholder', None) for _ in range(len(self.estimators)+1)]
        saved = [None for i in range(len(self.estimators))]
        scores = {}
        for current in pipelines:
            X_tst, X_trn = X_test, X_train
            name = tuple([fname]+[n for n, _ in current])
            total = 0
            try:
                for i, (pn, pt), (cn, ct) in zip(range(len(self.estimators)), prev[:-1], current[:-1]):
                    if pn != cn:
                        start = time.time()
                        X_trn = ct.fit_transform(X_trn, kwargs[cn] if cn in kwargs else y_train)
                        end = time.time()
                        X_tst = ct.transform(X_tst)
                        saved[i] = (end-start, X_trn, X_tst)
                        total += end-start
                    else:
                        t, X_trn, X_tst = saved[i]
                        total += t
                model = current[-1][1]
                start = time.time()
                model.fit(X_trn, y_train)
                total += time.time() - start
                scores[name] = self.score(model, X_tst, y_test, name=name)
                scores[name]['runtime'] = total
            except Exception:
                print('Error encoutered in scoring', name)
                raise
        return scores

    def cross_validate(self, y, test_sizes, folds, random_state=None):
        for fname, X in self.features.items():

            for split in test_sizes:
                cv = sklearn.model_selection.StratifiedShuffleSplit(
                    n_splits=folds, random_state=random_state, test_size=split)
                yield from zip(itertools.repeat(fname), itertools.repeat(X),
                               itertools.repeat(split), cv.split(X, y))

    def score(self, model, X, y, sample_weight=None, **kwargs):
        scores = {}
        for mname, metric in self.metrics.items():
            scores[mname] = metric(model, X, y)
        return scores


# via https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
def plot_learning_curve(estimator, X, y, axes=None, ylim=None, title=None, **kwargs):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        sklearn.model_selection.learning_curve(estimator, X, y, return_times=True, **kwargs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Data downloader')
    parser.add_argument('directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be stored')
    parser.add_argument('-s', '--subjects', type=str, nargs='+', help='A filter for which subjects to download data for. If not specified, run on all')
    parser.add_argument('-t', '--tasks', type=str, nargs='+', default=['rest'], help='Which tasks to download data for. By default, only download resting state data')
    parser.add_argument('-m', '--modalities', type=str, nargs='+', default=['anat', 'func'], help='Which measurement modalities to download. By default, download anatomical and fMRI data')
    parser.add_argument('-d', '--derivatives', type=str, nargs='*', default=['unprocessed'], help='Which derivative pipeline outputs to download. By default, download only the unprocessed data')
    parser.add_argument('--local', action='store_true', help='Only rely on already-downloaded files for preprocessing')
    parser.add_argument('--keep', action='store_true', help='Whether to delete the raw data files after preprocessing (NOT RECOMMENDED)')
    parser.add_argument('--override', action='store_true', help='Whether to download a file that matches a pattern that already exists on the file-system. The default behavior is to skip such files')
    parser.add_argument('-v', '--trace', action='store_true', help='Whether to output any commands that are being run. The default behavior is not to')
    parser.add_argument('-r', '--random', action='store_true', help='Whether to process the input files in random order')
    parser.add_argument('--progress', action='store_true', help='Whether to display a bar as an indicator of progress')
    parser.add_argument('-p', '--nproc', type=int, default=1, help='How many subprocesses to run the solver in')
    parser.add_argument('-n', '--limit', type=int, help='Stop after processing some number of inputs')
    parser.add_argument('--parcellations', type=int, nargs='*', help='Which parcellations to use. Otherwise, run on all')
    args = parser.parse_args()
    if args.nproc != 1 or args.parcellations:
        raise NotImplementedError

    logging.info('Setting up indices...')
    layout = get_layout(args.directory, args.derivatives)
    if args.subjects:
        subjects = args.subjects
    elif args.local:
        subjects = downloaded_subjects(layout)
    else:
        participants = pd.read_csv(os.path.join(args.directory, 'participants.tsv'), sep='\t')
        subjects = participants[[t for t in args.tasks if t in participants.columns]].any(axis=1)
        subjects = list(participants[subjects].participant_id)
    if args.random:
        random.shuffle(subjects)
    if args.limit:
        subjects = subjects[:args.limit]

    logging.info('Loading parcellations...')
    processor = Processor(args.derivatives)

    include_subjects_data(
        subjects, local=args.local, keep=args.keep, progress=args.progress, foreach=processor.process_callback, skip=processor.skip_callback, tasks=args.tasks,
        folder=args.directory, derivatives=args.derivatives, trace=args.trace, override=args.override, modalities=args.modalities)
