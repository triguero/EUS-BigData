package org.apache.mahout.keel.Algorithms.Sarah.AUC;

/**
*
* @author Sarah
*/
public class PosProb implements Comparable<PosProb>{
	
	// Indicates whether the instance belongs to the positive class
    private boolean isPositive ;
    
    // Score (probability of belonging to the positive class)
    private double prob;
    
    public PosProb (boolean isPositive , double prob){
        this.isPositive = isPositive;
        this.prob = prob;
    }
    
    public boolean isPositiveInstance(){
        return isPositive;
    }
    
    public double getProb (){
        return prob;
    }

    public int compareTo(PosProb o) {
        if (prob < o.getProb()){  // our element should be considered later
            return 1;
        } else if (prob > o.getProb()){
            return -1;
        } else {
            return 0;
        }
    }
    
    @Override
    public String toString(){
        String text = "( ";
        if(isPositive){
            text += "positive";
        } else {
            text += "negative";
        }
        return text + " , " + prob + " )" ;
    }

}
