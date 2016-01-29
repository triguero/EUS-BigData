package org.apache.mahout.keel.Algorithms.Sarah.AUC;

import java.util.Arrays;

/**
 *
 * @author Sarah
 */

public class CalculateAUC {

	/*
     * Method to calculate the Area Under The Curve (AUC)
     * @param probs: array of pairs, representing the actual class and the 
     * probability of belonging to the positive one     * 
     * @return the AUC
     */
    public static double calculate(PosProb[] probs){
        
        double auc = 0.0;
        
        if(probs != null){
                       
            /*
             * Elements are sorted in decreasing order of the probability of 
             * belonging to the positive class.
             */
            Arrays.sort(probs);

            /*
             * Count the number of positive and negative elements.
             */
            int pos = 0 ;
            int neg = 0 ;

            for (PosProb pair : probs) {
                if(pair.isPositiveInstance()){
                    pos++;
                } else {
                    neg++;
                }
            }

            /*
             * Obtain points of the ROC-curve, by stepwise lowering the 
             * threshold above which the instances are classified as positive.
             * The AUC is calculated by summing the areas of the trapezoids that
             * are created by connecting consecutive points of the ROC-curve by 
             * line segments.
             */
            if (pos != 0 && neg != 0){

                // Previous TPR and FPR (point on ROC-curve)
                double prevTPR = 0.0;
                double prevFPR = 0.0;
                double tp = 0.0;
                double fp = 0.0;
                double prevProb = Double.NEGATIVE_INFINITY;

               // Calculate AUC
                for(int j = 0; j < probs.length; j++){
                    PosProb el = probs[j];

                    double prob = el.getProb();
                    if(prob != prevProb){
                        double currTPR = tp / pos;
                        double currFPR = fp / neg;
                        double term = ((prevTPR + currTPR) * (currFPR - prevFPR)) / 2;
                        auc += term;
                        prevTPR = currTPR;
                        prevFPR = currFPR;
                        prevProb = prob;
                    }
                    if(el.isPositiveInstance()){
                        tp++;
                    } else {
                        fp++;
                    }
                }

                // Last point (1,1)
                auc += ((prevTPR + 1.0) * (1.0 - prevFPR)) / 2;

            }
        }
        
        return auc;
        
    }
}
