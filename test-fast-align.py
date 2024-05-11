import subprocess

# input_corpus_filepath = "fast_align_temp_file1.tmp"
# output_alignments_filepath = "fast_align_temp_file2.tmp"

# subprocess.run(
#     [
#         "./bin/fast_align",
#         "-i",
#         input_corpus_filepath,
#         "-d",
#         "-o",
#         "-v",
#         ">",
#         "test-output",
#     ]
# )

corpus_path = "fast_align_temp_file1.tmp"
out_path = "fast_align_temp_file2.tmp"

subprocess.run(f"./bin/fast_align -i {corpus_path} -d -o -v > {out_path}", shell=True)
