# coding: utf-8
"""
Play with Neurosynth database in NiMARE
"""
from datetime import datetime

import nimare

# Convert the (massive) database
print('{}: Converting Neurosynth dataset'.format(datetime.now()))
db_file = '/scratch/tsalo006/nimare-ohbm-2019/neurosynth-dataset/database.txt'
lb_file = '/scratch/tsalo006/nimare-ohbm-2019/neurosynth-dataset/features.txt'
nimare.io.convert_neurosynth_to_json(
    db_file,
    '/scratch/tsalo006/nimare-ohbm-2019/neurosynth-dataset/neurosynth_dset.json',
    annotations_file=lb_file)

print('{}: Loading Neurosynth dataset'.format(datetime.now()))
dset = nimare.dataset.Dataset(
    '/scratch/tsalo006/nimare-ohbm-2019/neurosynth-dataset/neurosynth_dset.json')

# Let's look at pain
pain_studies = dset.get_studies_by_label('pain', label_threshold=0.001)
print('{0}: There are {1} studies with the word "pain" in their '
      'abstract.'.format(datetime.now(), len(pain_studies)))

# Let's run a pain meta-analysis
# Neurosynth doesn't have sample sizes so we'll assume they all have 20
print('{}: Running ALE meta-analysis of pain'.format(datetime.now()))
sample_size = 20
ale = nimare.meta.cbma.ALE(dset, kernel__n=sample_size)
ale.fit(ids=pain_studies)
ale.results.save_results(output_dir='neurosynth_test/', prefix='pain')

# Now let's try functional decoding!
print('{}: Performing BrainMap-style decoding'.format(datetime.now()))
decoding_df = nimare.decode.discrete.brainmap_decode(
        dset.coordinates, dset.annotations, ids=pain_studies)
decoding_df.to_csv('neurosynth_test/pain_decoding.csv', index_label='Term')

print('{}: Workflow completed!'.format(datetime.now()))
