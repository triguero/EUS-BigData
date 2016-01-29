package org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles.C45;

import java.io.*;

import org.apache.mahout.keel.Dataset.Attributes;

public abstract class Algorithm {
    /** The name of the file that contains the information to build the model. */
    protected static String modelFileName = "";

    /** The name of the file that contains the information to make the training. */
    protected static String trainFileName = "";

    /** The name of the file that contains the information to make the test. */
    protected static String testFileName = "";

    /** The name of the train output file. */
    protected static String trainOutputFileName;

    /** The name of the test output file. */
    protected static String testOutputFileName;

    /** The name of the result file. */
    protected static String resultFileName;

    /** Correctly classified itemsets. */
    protected int correct = 0;

    /** Correctly classified in test. */
    protected int testCorrect = 0;

    /** The model dataset. */
    protected Dataset modelDataset;

    /** The train dataset. */
    protected Dataset trainDataset;

    /** The test dataset. */
    protected Dataset testDataset;

    /** The log file. */
    protected static BufferedWriter log;

    /** The instant of starting the algorithm. */
    protected long startTime = System.currentTimeMillis();

    /** Function to initialize the stream tokenizer.
     *
     * @param tokenizer		The tokenizer.
     */
    protected void initTokenizer(StreamTokenizer tokenizer) {
        tokenizer.resetSyntax();
        tokenizer.whitespaceChars(0, ' ');
        tokenizer.wordChars(' ' + 1, '\u00FF');
        tokenizer.whitespaceChars(',', ',');
        tokenizer.quoteChar('"');
        tokenizer.quoteChar('\'');
        tokenizer.ordinaryChar('=');
        tokenizer.ordinaryChar('{');
        tokenizer.ordinaryChar('}');
        tokenizer.ordinaryChar('[');
        tokenizer.ordinaryChar(']');
        tokenizer.eolIsSignificant(true);
    }


    /** Function to get the name of the relation and the names, types and possible values of every attribute in
     *  a dataset.
     *
     * @return The name and the attributes of the relation.
     */
    protected String getHeader() {
        String header;
        header = "@relation " + Attributes.getRelationName() + "\n";
        header += Attributes.getInputAttributesHeader();
        header += Attributes.getOutputAttributesHeader();
        header += Attributes.getInputHeader() + "\n";
        header += Attributes.getOutputHeader() + "\n";
        header += "@data\n";

        return header;
    }


    /** Puts the tokenizer in the first token of the next line.
     *
     * @param tokenizer		The tokenizer which reads this function.
     *
     * @return				True if reaches the end of file. False otherwise.
     *
     * @throws Exception	If cannot read the tokenizer.
     */
    protected boolean getNextToken(StreamTokenizer tokenizer) throws Exception {
        try {
            if (tokenizer.nextToken() == StreamTokenizer.TT_EOF) {
                return false;
            } else {
                tokenizer.pushBack();
                while (tokenizer.nextToken() != StreamTokenizer.TT_EOL) {
                    ;
                } while (tokenizer.nextToken() == StreamTokenizer.TT_EOL) {
                    ;
                }

                if (tokenizer.sval == null) {
                    return false;
                } else {
                    return true;
                }
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
            return false;
        }
    }


    /** Function to read the options from the execution file and assign the values to the parameters.
     *
     * @param options 		The StreamTokenizer that reads the parameters file.
     *
     * @throws Exception	If the format of the file is not correct.
     */
    protected abstract void setOptions(StreamTokenizer options) throws
            Exception;

    /** Evaluates the algorithm and writes the results in the file.
     *
     * @exception 	If the file cannot be written.
     */
    protected abstract void printResult() throws IOException;

    /** Evaluates the test dataset and writes the results in the file.
     *
     * @exception 	If the file cannot be written.
     */
    protected abstract void printTest() throws IOException;

    /** Evaluates the training dataset and writes the results in the file.
     *
     * @exception 	If the file cannot be written.
     */
    protected abstract void printTrain() throws IOException;
}
