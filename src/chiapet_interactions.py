import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ChiaPetInteractions(object):

    def __init__(self, path, bin_length, chromosomes=None, different_chrs=False):
        self.bed_file = pd.read_csv(path, sep='\t', lineterminator='\n', header=None,
                                    names=['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'thickStart',
                                           'thickEnd',
                                           'itemRgb', 'blockCount', 'blockSizes', 'blockStarts'])


        self.bin_length = bin_length
        bed_file = self.bed_file

        bed_file.drop_duplicates('name', inplace=True)

        if not chromosomes:
            self.chromosomes = bed_file['chrom'].unique()
        else:
            if not isinstance(chromosomes, list):
                chromosomes = [chromosomes]
            self.chromosomes = chromosomes

        self.interactions = self.__bed2interactions(chromosomes, different_chrs)
        self.bounds = self.__get_bounds()
        self.interactions, self.offsets = self.__apply_offsets()

        total_length = max(self.interactions['index2'])
        n_bins = total_length // bin_length

        print("Number of bins:", n_bins)
        print("Number of interactions:", np.count_nonzero(self.interactions))

        self.heatmap = None

    def __bed2interactions(self, chromosomes, different_chrs):
        """ Returns a DataFrame with the chromosomes and indices of interactions"""
        bed = pd.DataFrame(columns=['chrom1', 'chrom2', 'index1', 'index2'])

        # Extract interactions only from the name
        # Because if the interaction is between different chromosomes,
        # Then the second chromosome is mentioned only in the name

        # Remove ",number" at the end of the name
        names = self.bed_file['name'].str.split(',').str[0]
        # Split the name into the two segments
        name_split = names.str.split('-', n=2, expand=True)
        segment_1 = name_split[0]
        segment_2 = name_split[1]

        segment_1_split = segment_1.str.split(':', n=2, expand=True)
        segment_2_split = segment_2.str.split(':', n=2, expand=True)

        chr_1 = segment_1_split[0]
        chr_2 = segment_2_split[0]
        contact_pos_1 = pd.to_numeric(segment_1_split[1].str.split('.').str[0])
        contact_pos_2 = pd.to_numeric(segment_2_split[1].str.split('.').str[-1])

        if different_chrs:
            different_chrs_selection = chr_1 != chr_2
            chr_1 = chr_1[different_chrs_selection]
            chr_2 = chr_2[different_chrs_selection]
            contact_pos_1 = contact_pos_1[different_chrs_selection]
            contact_pos_2 = contact_pos_2[different_chrs_selection]

        if chromosomes:
            chromosomes_selection = np.isin([chr_1, chr_2], chromosomes, assume_unique=True)

            # Select rows only if both the chromosome regions are in the chromosomes list
            chromosomes_selection = np.logical_and(chromosomes_selection[0], chromosomes_selection[1])
            chr_1 = chr_1[chromosomes_selection]
            chr_2 = chr_2[chromosomes_selection]
            contact_pos_1 = contact_pos_1[chromosomes_selection]
            contact_pos_2 = contact_pos_2[chromosomes_selection]

        bed['chrom1'] = np.hstack((chr_1, chr_2))
        bed['index1'] = np.hstack((contact_pos_1, contact_pos_2))

        bed['chrom2'] = np.hstack((chr_2, chr_1))
        bed['index2'] = np.hstack((contact_pos_2, contact_pos_1))
        print(len(bed[bed['chrom1'] == bed['chrom2']]))
        return bed

    def __get_bounds(self):
        """ Returns the lowest and highest positions of contact
        for any given chromosome"""
        bounds = dict()
        for chromosome in self.chromosomes:
            interactions_chr1 = self.interactions[self.interactions['chrom1'] == chromosome]['index1']
            interactions_chr2 = self.interactions[self.interactions['chrom2'] == chromosome]['index2']
            if len(interactions_chr1) == 0 or len(interactions_chr2) == 0:
                bounds[chromosome] = (0, 0)
            else:
                interactions_chr = np.hstack((interactions_chr1, interactions_chr2))
                bounds[chromosome] = (min(interactions_chr), max(interactions_chr))
        return bounds

    def __apply_offsets(self):
        """ Returns:
        1. the interactions with positions aligned
        to be aligned in a heatmap according to the chromosomes.
        2. the start indices of each chromosome in the new interactions"""
        total_offset = 0
        offsets = dict()
        for chromosome in self.chromosomes:
            interactions_1 = self.interactions.loc[self.interactions['chrom1'] == chromosome, 'index1']
            interactions_2 = self.interactions.loc[self.interactions['chrom2'] == chromosome, 'index2']
            start_offset = self.bounds[chromosome][0]
            end_offset = self.bounds[chromosome][1]
            self.interactions.loc[
                self.interactions['chrom1'] == chromosome, 'index1'] = interactions_1 - start_offset + total_offset
            self.interactions.loc[
                self.interactions['chrom2'] == chromosome, 'index2'] = interactions_2 - start_offset + total_offset
            offsets[chromosome] = total_offset
            total_offset += end_offset + 1
        return self.interactions, offsets

    def generate_heatmap(self):
        end = max(self.interactions['index2'])
        bins = np.arange(0, end, self.bin_length)
        self.heatmap, _, _ = np.histogram2d(self.interactions['index1'], self.interactions['index2'], bins)
        return self.heatmap

    def remove_top_outliers(self, ratio=0.05):
        if self.heatmap is None:
            self.generate_heatmap()

        top = int(np.count_nonzero(self.heatmap) * ratio)
        print("Remove top", top, "elements")
        # Remove the top outliers
        self.heatmap[
            np.unravel_index([np.argpartition(-self.heatmap, (0, top), axis=None)[:top]], self.heatmap.shape)] = 0
        return self.heatmap

    def plot_heatmap(self):
        if self.heatmap is None:
            self.generate_heatmap()

        plt.clf()
        plt.imshow(self.heatmap.T, cmap='Greys')
        plt.show()


def main():
    bin_length = 100000
    file_path = '../ENCFF000KYD_K562_CTCF.bed'

    contact_matrix = ChiaPetInteractions(file_path, bin_length, chromosomes=['chr1'], different_chrs=False)
    contact_matrix.generate_heatmap()
    contact_matrix.remove_top_outliers(ratio=0.005)
    contact_matrix.plot_heatmap()


if __name__ == '__main__':
    main()
