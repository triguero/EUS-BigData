package org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles.C45;


/**
 * Class to manipulate an itemset.
 */
public class Itemset {
    /** The dataset which the itemset has access to. */
    protected Dataset dataset;

    /** Values of the itemset. */
    protected double[] values;

    /** The weight of the itemset. */
    protected double weight;

    /** Constant that represents the missing value. */
    protected final static double MISSING_VALUE = Double.NaN;

    /** Constructor that copies the values and the weight.
     *
     * @param itemset		The itemset to copy.
     */
    public Itemset(Itemset itemset) {
        values = itemset.values;
        weight = itemset.weight;
        dataset = null;
    }

    /** Constructor that sets the values and the weight.
     *
     * @param w					The weight.
     * @param attributeValues	The values.
     */
    public Itemset(double w, double[] attributeValues) {
        values = attributeValues;
        weight = w;
        dataset = null;
    }

    /** Returns the index of the class attribute.
     *
     */
    public int classIndex() {
        if (dataset == null) {
            //throw new RuntimeException( "Itemset does not have access to a dataset." );
            System.err.println("dataset in itemset is null");
            return ( -1);
        } else {
            return dataset.getClassIndex();
        }
    }

    /** Function to test if the class attribute is missing.
     *
     * @return	True if the value of the class attribute is missing.
     */
    public boolean classIsMissing() {
        if (classIndex() < 0) {
            throw new RuntimeException("Class is not set.");
        } else {
            return isMissing(classIndex());
        }
    }

    /** Returns the index of the value of the class.
     *
     */
    public double getClassValue() {
        if (classIndex() < 0) {
            System.err.println("dataset in itemset is null");
            return ( -1);
        } else {
            return getValue(classIndex());
        }
    }

    /** Returns the number of class values.
     *
     */
    public int numClasses() {
        if (dataset == null) {
            System.err.println("dataset in itemset is null");
            return ( -1);
        } else {
            return dataset.numClasses();
        }
    }

    /** Returns the attribute with the given index.
     *
     */
    public Attribute getAttribute(int index) {
        if (dataset == null) {
            System.err.println("dataset in itemset is null");
            return null;
        } else {
            return dataset.getAttribute(index);
        }
    }

    /** Function to set a value.
     *
     * @param index		The index of the attribute.
     * @param value		The value.
     */
    public void setValue(int index, double value) {
        double[] help = new double[values.length];

        System.arraycopy(values, 0, help, 0, values.length);
        values = help;
        values[index] = value;
    }

    /** Returns the value of the given attribute.
     *
     */
    public double getValue(int index) {
        return values[index];
    }

    /** Function to set the weight.
     *
     * @param w		The weight.
     */
    public final void setWeight(double w) {
        weight = w;
    }

    /** Returns the itemset weight.
     *
     */
    public final double getWeight() {
        return weight;
    }

    /** Returns the dataset of this itemset.
     *
     */
    public Dataset getDataset() {
        return dataset;
    }

    /** Function to set the dataset.
     *
     * @param data	The dataset.
     */
    public final void setDataset(Dataset data) {
        dataset = data;
    }

    /** Function to check if a value is missing.
     *
     * @param index	The index of the attribute to check.
     *
     * @return		True is the value of the attribute is missing. False otherwise.
     */
    public boolean isMissing(int index) {
        if (Double.isNaN(values[index])) {
            return true;
        } else {
            return false;
        }
    }

    /** Function to check if the value given is the missing value.
     *
     * @param val	The value to check.
     *
     * @return		True if the value given is the missing value. False otherwise.
     */
    public static boolean isMissingValue(double val) {
        return Double.isNaN(val);
    }

    /** Returns the missing value.
     *
     */
    public static double getMissingValue() {
        return MISSING_VALUE;
    }

    /** Function to set as missing the class value.
     *
     */
    public void setClassMissing() {
        if (classIndex() < 0) {
            throw new RuntimeException("Class is not set.");
        } else {
            setMissing(classIndex());
        }
    }

    /** Function to set a value as missing.
     *
     * @param index	The index of the attribute.
     */
    public final void setMissing(int index) {
        setValue(index, MISSING_VALUE);
    }

    /** Function to copy an itemset.
     *
     * @return	The itemset created.
     */
    public Object copy() {
        Itemset result = new Itemset(this);
        result.dataset = dataset;

        return result;
    }

    /** Function to print the itemset.
     *
     */
    public String toString() {
        String result = "";
        for (int i = 0; i < dataset.numAttributes(); i++) {
            Attribute att = dataset.getAttribute(i);

            if (att.isContinuous()) {
                result += att.name() + "=" + values[i] + "\n";
            } else {
                result += att.name() + "=" + att.value((int) values[i]) + "\n";
            }
        }

        return result;
    }
}
