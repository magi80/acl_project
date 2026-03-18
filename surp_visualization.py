import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys


def visualize_surp_data(word_surp, sentence_surp, hist=False, kde=False, line_plot=False):
    data_word = pd.read_csv(word_surp, sep='\t')
    data_sent= pd.read_csv(sentence_surp, sep='\t')

    # Preparing word counts in y-axis for lineplot
    data_word['freq'] = data_word.groupby(['word', 'context_window'])['word'].transform('size')
    data_sent['freq'] = data_sent.groupby(['word', 'context_window'])['word'].transform('size')

    #print(data_word[['word', 'freq', 'context_window']].head(50))
    #print(data_sent[['word', 'freq', 'context_window']].head(50))

    if kde: 
        sns.kdeplot(data=data_word, x='surprisal', hue='context_window')
        plt.xlabel('Surprisal')
        plt.ylabel('Frequency')
        plt.title('Cross Surprisal Word Level - KDE')
        #plt.savefig('csw.png', dpi=300)
        plt.show()

        sns.kdeplot(data=data_sent, x='surprisal', hue='context_window')
        plt.xlabel('Surprisal')
        plt.ylabel('Frequency')
        plt.title('Cross Surprisal Sentence Level - KDE')
        #plt.savefig('css.png', dpi=300)
        plt.show()

    if hist:
        w = sns.FacetGrid(data=data_word, col='context_window', col_wrap=3, height=3, hue='context_window')
        w.map_dataframe(sns.histplot, x='surprisal', binwidth=2, palette='flare')
        w.set_axis_labels('Surprisal', 'Frequency')
        plt.suptitle('Surprisal Distribution by Context Window - Word Level ')
        plt.tight_layout()
        plt.show()

        s = sns.FacetGrid(data=data_sent, col='context_window', col_wrap=2, height=3, hue='context_window')
        s.set_axis_labels('Surprisal', 'Frequency')
        s.map_dataframe(sns.histplot, x='surprisal', binwidth=2)
        plt.suptitle('Surprisal Distribution by Context Window - Sentence Level ')
        plt.tight_layout()
        plt.show()

    if line_plot:
        w = sns.FacetGrid(data=data_word, col='context_window', col_wrap=3, height=3, aspect=1.5)
        w.map_dataframe(sns.lineplot,  x='surprisal', y='freq', linewidth=1, markersize=4,
              dashes=False,  style='context_window', markers=True)
        w.set_axis_labels('Surprisal', 'Frequency')
        plt.suptitle('Surprisal Distribution by Context Window - Word Level ')
        plt.tight_layout()
        plt.show()

        s = sns.FacetGrid(data=data_sent, col='context_window', col_wrap=2, height=3, aspect=1.5)
        s.map_dataframe(sns.lineplot, x='surprisal', y='freq', linewidth=1, markersize=4,
              dashes=False, style='context_window', markers=True)
        s.set_axis_labels('Surprisal', 'Frequency')
        plt.suptitle('Surprisal Distribution by Context Window - Sentence Level ')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':  
    word_level = sys.argv[1] # saved .csv file, words
    sent_level = sys.argv[2] # saved .csv file, sentences
    visualize_surp_data(word_level, sent_level, kde=True, hist=True, line_plot=True)