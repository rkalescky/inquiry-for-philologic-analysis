import luigi


class preprocess(luigi.Task):

    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget("output.tsv")

    def run(self):
        # run preprocess.py functions


class lda(luigi.Task):

    def requires(self):
        return [preprocess()]

    def output(self):
        return luigi.LocalTarget("malletfile.txt")

    def run(self):
        # run mallet using subprocess


class rankDebates(luigi.Task):

    def requires(self):
        return [lda()]

    def output(self):
        return luigi.LocalTarget("ranked_debates.tsv")

    def run(self):
        # rank_debates.py functions


if __name__ == '__main__':
    luigi.run()
