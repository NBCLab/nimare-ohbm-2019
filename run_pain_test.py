# coding: utf-8
"""
Play with Neurosynth database in NiMARE
"""
from os import mkdir
import os.path as op
from datetime import datetime

import nimare
from nimare import decode

in_dir = '/scratch/tsalo006/nimare-ohbm-2019/'
in_dir = '/Users/tsalo/Documents/nbc/nimare-ohbm-2019/'
out_dir = op.join(in_dir, 'neurosynth-test')
if not op.isdir(out_dir):
    mkdir(out_dir)

# Convert the (massive) database
print('{}: Converting Neurosynth dataset'.format(datetime.now()))
#db_file = op.join(in_dir, 'neurosynth-dataset/database.txt')
#lb_file = op.join(in_dir, 'neurosynth-dataset/features.txt')
#nimare.io.convert_neurosynth_to_json(
#    db_file,
#    op.join(in_dir, 'neurosynth-dataset/neurosynth_dset.json'),
#    annotations_file=lb_file)

print('{}: Loading Neurosynth dataset'.format(datetime.now()))
dset = nimare.dataset.Dataset(
    op.join(in_dir, 'neurosynth-dataset/neurosynth_dset.json'))
print(len(dset.ids))

# Let's look at pain
pain_studies = dset.get_studies_by_label('pain', label_threshold=0.001)
print('{0}: There are {1} studies with the word "pain" in their '
      'abstracts.'.format(datetime.now(), len(pain_studies)))
pain_dset = dset.slice(ids=pain_studies)

# Now let's try functional decoding!
print('{}: Performing Neurosynth-style decoding'.format(datetime.now()))
decoding_df = decode.discrete.neurosynth_decode(
        dset.coordinates, dset.annotations, ids=pain_studies)
decoding_df.to_csv(op.join(out_dir, 'pain_decoding_neurosynth.csv'), index_label='Term')

# Now let's try functional decoding!
print('{}: Performing BrainMap-style decoding'.format(datetime.now()))
decoding_df = decode.discrete.brainmap_decode(
        dset.coordinates, dset.annotations, ids=pain_studies)
decoding_df.to_csv(op.join(out_dir, 'pain_decoding_brainmap.csv'), index_label='Term')

# Let's run a pain meta-analysis
# Neurosynth doesn't have sample sizes so we'll assume they all have 20
print('{}: Running ALE meta-analysis of pain'.format(datetime.now()))
sample_size = 20
ale = nimare.meta.cbma.ALE(dset, kernel__n=sample_size)
ale.fit(ids=pain_studies, n_iters=100)
ale.results.save_results(output_dir=out_dir, prefix='pain')

print('{}: Workflow completed!'.format(datetime.now()))
