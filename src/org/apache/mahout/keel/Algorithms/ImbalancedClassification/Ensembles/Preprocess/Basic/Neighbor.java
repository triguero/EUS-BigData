/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles.Preprocess.Basic;

/**
 *
 * @author mikel
 */
public class Neighbor implements Comparable {
    int index;
    double distance;
    
    public Neighbor(int index, double distance) {
        this.index = index;
        this.distance = distance;
    }
    
    
    public int getIndex() {
        return index;
    }
    
    public double getDistance() {
        return distance;
    }
    
    public void setIndex(int index) {
        this.index = index;
    }
    
 //   @Override
    public int compareTo (Object n1) {
        if ( ((Neighbor)n1).getDistance() < this.getDistance())
            return 1;
        else if (((Neighbor)n1).getDistance() > this.getDistance())
            return -1;
        else 
            return 0;
    }
    
    @Override
    public boolean equals(Object n1)
    {
        if (((Neighbor)n1).index == this.index)
            return true;
        else
            return false;
    }
}
