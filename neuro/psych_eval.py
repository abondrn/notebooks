def mixture_pca(X, n_mixtures):
    cls = sklearn.cluster.SpectralClustering(affinity='precomputed', n_clusters=n_mixtures)
    pca = sklearn.decomposition.PCA(n_components=2, whiten=True)
    mixture_labels = cls.fit_predict(np.corrcoef(X.T))
    mixtures = []
    residuals = []
    for mixture in range(n_mixtures):
        sel = mixture_labels == mixture
        embedding = np.eye(X.shape[1])[:, sel].dot(pca.fit_transform(X[:, sel].T)).T
        mixtures.append(embedding[0])
    return mixture_labels, np.array(mixtures)

cls, vecs = mixture_pca(psych.values, 8)
list(zip(psych.columns, cls))

---

plt.matshow(vecs.T)
plt.colorbar()
plt.yticks(range(len(psych.columns)), psych.columns, rotation=45);

---

X_r = sklearn.decomposition.PCA(2).fit_transform(psych.values.dot(std.T))
for cond, color in zip(conditions, colors):
    plt.scatter(X_r[participant_metadata.diagnosis == cond, 0], X_r[participant_metadata.diagnosis == cond, 1], color=color, alpha=.8, lw=2,
                label=cond)
    plt.legend()

'''
fcMRI results have been particularly useful for understanding how distributed networks involving association cortex are organized (Supplementary Video 1). Human association cortex is vastly expanded relative to the monkey21, and it is unclear how putatively cognitive systems are situated in rela-tion to each other and to limbic systems. One landmark observation was that a distributed network of association regions, often referred to as the default network, behaves as a functionally coupled system3.
> fcMRI  measures  are  also  sensitive  to  confounding  factors  that include head motion27,28 and physiological artifacts linked to respi-ratory and cardiac rhythms29. Methods for dealing with these factors are being explored by many laboratories, but a concern is that many published results pertaining to individual and group differences are artifacts of either head motion or actual neural events associated with motion and breathing.
> Second, the functional regions active during rest parallel those regions active dur-ing tasks that require subjects to engage in internally directed mental operations32. Thus, rest states may involve task-dependent coactivation of regions, much like any other experimentally controlled task. Finally, although functional connectivity patterns derived from rest are good predictors of the organization of task-based patterns33, they are not better predictors than other functional connectivity patterns derived from task states
> Mild forms of anxiety or low mood may be associated with systematic differences in sponta-neous mental events35. An anecdotal observation is that adults with autism can sometimes comply more vigilantly with instructions to fixate and hold still than typical adults. Although the implications of this observation are unclear, it is a reminder to leave open the pos-sibility that observed differences in functional connectivity may be related to individual differences in transient task- or state-dependent factors,  in  addition  to  underlying  differences  in  stable  features  of brain organization.
> (a) Functional connectivity suggests that the default network is negatively coupled (anticorrelated) to brain networks that are used for focused external visual attention50  ,51. Anticorrelated networks are displayed by plotting those regions that negatively correlate with the default network in blue in addition to positive correlations in yellow–orange. (b) A correlation matrix shows the complete coupling architecture of the full cerebral cortex measured at rest. Regions fall in the networks labeled in Figure 2, as well as a limbic network from ref. 37. SomMot, somatomotor; DorsAttn, dorsal attention. Between-network correlations are characterized by both positive and negative relations, with strong anticorrelation notable between the default and salience/dorsal attention networks
> Taking advantage of this property, two studies in 2005 reported that spontaneous fluctuations in the dorsal attention system are strongly anticorrelated with those of  the  default  network52,53.  As  normalized  functional  MRI  signals increase in the dorsal attention system, they decrease in the default network  (Fig. 5).  This  discovery  may  mark  a  fundamental  feature of brain organization that had not been appreciated by earlier tech-niques.  The  dorsal  attention  system  is  associated  with  processing information from external sensory channels; the default network is characterized by processing of internally focused information, such as during remembering or mentally imagining the future.
> The problem with interpreting anticorrelations arises because fcMRI is a relative measure. It depends on signal change estimates that are extracted after processing and denoising steps have been applied. Owing to the way fcMRI data are typically normalized, it is difficult to surmise how to interpret the meaning of the correlation strength and whether the sign of correlation should be interpreted at all55. New strategies for processing fcMRI data may mitigate the specific issue of  normalization56.  But  the  ambiguity  speaks  directly  to  a  broader limitation  of  fcMRI:  fcMRI  is  difficult  to  interpret  because  it  is  an indirect, relative measure of neural activity fluctuations. As a result, observations such as anticorrelations are ambiguous without more insight into their mechanisms.

> Nevertheless, the validity of individual differences measured in the common space isnot assured because individual differences of interest may also be mixed into the transformationmatrix for each subject. We would encourage[11_TD$DIFF]researchers to explore hyperalignment for theirdatasets, and to report results obtained with and without hyperalignment, such that we canaccumulate further evidence for its most appropriate application. In general, we would recom-mend that investigators try more than one approach to alignment, and report all of them, so wecan see which might work best for which kinds of questions.
> Calibrated fMRI: (e.g., requiring concurrent measurement of BOLD and CBF and inhalation of CO2[63,64]) and isstill imperfect[65].
> Normalization: For example, the BOLD response tohypercapnia, induced through administration of CO2[66]or by using a breath-hold challenge[67], can be used as a normalization factor (Figure 3A). Alternatively, whole-brain venousoxygenation levels can be measured with a special pulse sequence and used to normalizethe BOLD response[68]. A more easily applicable option is to use the amplitude of low-frequencyfluctuations in resting-state fMRI data (RS-ALFF)[69,70]as a normalization factor;indeed RS-ALFF reflects naturally-occurring variations in cardiac rhythm and in respiratory rateand depth[71], and approximates the BOLD response to a hypercapnic challenge (Figure 3A). Infact, one does not even need to acquire a separate resting-state scan. In the same way thatfunctional connectivity can be derived from the residuals of ageneral linear model(GLM) fortask-based fMRI data[72], the amplitude of low-frequencyfluctuations in the residuals of task-based fMRI data (GLMres-ALFF) can also be used to rescale the BOLD signal change; this‘vascular autorescaling’(VasA) technique was even shown to outperform RS-ALFF-basednormalization[73](Figure 3B).
> Test–retest reliability quantifies how variable the established relationship is in the samesample of subjects, under the same conditions (stimulus, scanner, time of day, analysis) at anappropriate time-interval (Figure 4A). It is also important to ensure that the relationship is robustto the exact preprocessing performed on the raw data, which can be conceived of as‘inter-rater’reliability (Figure 4B). In addition, because we are interested in brain function rather than the low-level properties of a given stimulus, the relationship must also be robust to the exact experi-mental conditions, known in psychology as‘parallel forms’reliability (Figure 4C). The relationshipshould further hold for a different sample of subjects (from the same population) (Figure 4D), andacross scanners.
> Though many measures of reliability have been proposed over the years[75], some directlyaddressing the ratio of inter- to intra-subject variance[83], a predictive framework inspired bymachine-learning is best suited to establish the reliability needed for individual differencesresearch, as we discuss further below (see‘Choosing Prediction over Correlation’).
> Sources of Within-Subject Variance We Acknowledge and Strive To Correct ForThere are several sources of within-subject, inter-session variance that are well known, andwhich fMRI researchers strive to eliminate at acquisition time and/or correct for during pre-processing of the data[84,85]. Scanner-related noise, artifacts, and drift are unavoidable but arefairly simple to address through artifact rejection and temporalfiltering. We briefly review heretwo main sources of within-subject variance which are more problematic and the state-of-the-artin addressing them: subject motion and subject body physiology.
> Motion arguably contributes more to inter-subjectvariance than to intra-subject variance (e.g., men tend to exhibit more head movements thanwomen[90], older people move more than younger people[94], and people with autism movemore than controls[95]). Motion artifacts have complex effects on fMRI statistics, and incom-pletely correcting for them can lead to erroneous conclusions in individual differences research[86,95–97].

Why? This boils down to a few problems inherit to functional brain imaging:
 - Resting state data is noisy, averaging groups of “similar” voxels reduces the effect of random noise effects
 - Provide an interpretative framework to functional imaging data. For example one parcellation group might be defined as the Default Mode Network which is thought to be functionally significant. So averaging voxels together belonging to the Default Mode Network provides an average estimate of the Default Mode Network signal. In addition the discovery of the Default Mode Network has yielded important insights into the organizational principles of the brain.
 - Limit the number of statistical tests thereby reducing potential Type I errors without resorting to strong statistical correction techniques that might reduce statistical power.
 - A simpler way to visualize your data, instead of 40x40x40=6400 data points, you might have 17 or up to 200; this is still significantly less data to deal with!

 - A key feature of the Yeo2011 networks is that they are spatially distributed, meaning that the locations of two voxels in the same network need not be part of the same region. 
 - Parcellations group voxels based on criteria such as similarities, orthogonality or some other criteria
 - Parcellations are defined by assigning each voxel a parcel ‘membership’ value telling you which group the parcel belongs to
 - Parcellations provide an interpretative framework for understanding resting state data. But beware, some of the techniques used to form parcellations may not represent actual brain functional units!
'''