package org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles;


/**
 * This class represents pairs holding the class prediction and value of the voting procedure.
 * It is used for AUC computations.
 * 
 * @author Sarah
 */
public class PredPair {
	
    private String prediction;
    private double sum;
    
    public PredPair(String pred, double sum){
        prediction = pred;
        this.sum = sum;
    }
    
    public String getPrediction(){
        return prediction;
    }    
    
    public double getVotingValue(){
        return sum;
    }

}
