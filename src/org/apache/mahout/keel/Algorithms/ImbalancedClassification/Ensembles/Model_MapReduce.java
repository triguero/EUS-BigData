/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutput;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;



/**
 *
 * @author mikel
 */
public class Model_MapReduce  implements java.io.Serializable {
    
	public String ensembleType; 
	public int n_classifiers;
	public boolean valid[];
	public RuleBase[] treeRuleSet; 
	public double[] alfa;
        

	
        public Model_MapReduce() {
            ensembleType = "";
            n_classifiers = 0;
            valid = null;
            treeRuleSet = null;
            alfa = null;
        }
        
        public Model_MapReduce(String ensembleType, int n_classifiers, boolean[] valid, RuleBase[] treeRuleSet, double[] alfa) {
            this.ensembleType = ensembleType;
            this.n_classifiers = n_classifiers;
            this.valid = valid.clone();
            this.treeRuleSet = treeRuleSet.clone(); // clone malo, soy consciente
            this.alfa = alfa.clone();
            
        }
        

        
        public void writeModel(String fileName) {
            try
            {
               FileOutputStream fileOut =
               new FileOutputStream(fileName);
               ObjectOutputStream out = new ObjectOutputStream(fileOut);
               out.writeObject(this);                          
               
               out.close();
               fileOut.close();
               System.out.println("Serialized data is saved in " + fileName);
            }catch(IOException i) {
                i.printStackTrace();
            }
        }
        
        public byte[] writeModel() {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            try
            {
               ObjectOutputStream out = new ObjectOutputStream(baos);
               out.writeObject(this);                          
               
              
               out.close();
               baos.close();
               
            }catch(IOException i) {
                i.printStackTrace();
            }
            
            return baos.toByteArray();
        }
        
        
        public void readModel(String fileName) {
            try
            {
               FileInputStream fileIn = new FileInputStream(fileName);
               ObjectInputStream in = new ObjectInputStream(fileIn);
               Model_MapReduce aux = (Model_MapReduce) in.readObject();
               
               

               treeRuleSet = aux.treeRuleSet;
               alfa = aux.alfa;
               valid = aux.valid;
               ensembleType = aux.ensembleType;
               n_classifiers = aux.n_classifiers;

               in.close();
               fileIn.close();
            }catch(IOException i)
            {
               i.printStackTrace();
               return;
            } catch(ClassNotFoundException c) {
               System.out.println("Model_MapReduce class not found");
               c.printStackTrace();
               return;
            }
        }
        
        
        public void readModel(byte[] byteArray) {
            try
            {
               ByteArrayInputStream bais = new ByteArrayInputStream(byteArray);
               ObjectInputStream in = new ObjectInputStream(bais);
               Model_MapReduce aux = (Model_MapReduce) in.readObject();
               
               

               treeRuleSet = aux.treeRuleSet;
               alfa = aux.alfa;
               valid = aux.valid;
               ensembleType = aux.ensembleType;
               n_classifiers = aux.n_classifiers;

               System.out.println("numero de classifiers:"+n_classifiers);
               in.close();
               bais.close();
            }catch(IOException i)
            {
               i.printStackTrace();
               return;
            } catch(ClassNotFoundException c) {
               System.out.println("Model_MapReduce class not found");
               c.printStackTrace();
               return;
            }
        }
        
        
        
        public void addModel(byte[] byteArray) {
            
            if (n_classifiers == 0) {
                readModel(byteArray);
                return;
            }
            
            
            Model_MapReduce new_model = new Model_MapReduce();
            new_model.readModel(byteArray);     
            //ensembleType = new_model.ensembleType;
            
            int offset = n_classifiers;
            n_classifiers += new_model.n_classifiers;
            alfa = Arrays.copyOf(alfa, n_classifiers);
            valid = Arrays.copyOf(valid, n_classifiers);
            treeRuleSet = Arrays.copyOf(treeRuleSet, n_classifiers);
            


            
            System.arraycopy(new_model.alfa, 0, alfa, offset, new_model.n_classifiers);
            System.arraycopy(new_model.valid, 0, valid, offset, new_model.n_classifiers);
            System.arraycopy(new_model.treeRuleSet, 0, treeRuleSet, offset, new_model.n_classifiers);
            
            
        }

/**
This method is used in the Spark version, in which we DON'T write the model as byte.. we directly aggregate 
*/
        public void addModel(Model_MapReduce new_model) {
            
            
            if (n_classifiers == 0) {
               treeRuleSet = new_model.treeRuleSet;
               alfa = new_model.alfa;
               valid = new_model.valid;
               ensembleType = new_model.ensembleType;
               n_classifiers = new_model.n_classifiers;
               
		return;
            }
            

        
            int offset = n_classifiers;
            n_classifiers += new_model.n_classifiers;
            alfa = Arrays.copyOf(alfa, n_classifiers);
            valid = Arrays.copyOf(valid, n_classifiers);
            treeRuleSet = Arrays.copyOf(treeRuleSet, n_classifiers);
            
          
            System.arraycopy(new_model.alfa, 0, alfa, offset, new_model.n_classifiers);
            System.arraycopy(new_model.valid, 0, valid, offset, new_model.n_classifiers);
            System.arraycopy(new_model.treeRuleSet, 0, treeRuleSet, offset, new_model.n_classifiers);
            
            
        }
        
        public void readModels(String[] fileNames) {
            
            Model_MapReduce[] models = new Model_MapReduce[fileNames.length];
            
            for (int i = 0; i < fileNames.length; i++) {   
                models[i] = new Model_MapReduce();
                models[i].readModel(fileNames[i]);      
                n_classifiers += models[i].n_classifiers;          
            }
            
            ensembleType = models[0].ensembleType;
            // los fusiono, alfa, treeRuleSet y valid
            alfa = new double[n_classifiers];
            valid = new boolean[n_classifiers];
            treeRuleSet = new RuleBase[n_classifiers];
            int offset = 0;
            for (int i = 0; i < fileNames.length; i++) {  
                System.arraycopy(models[i].alfa, 0, alfa, offset, models[i].n_classifiers);
                System.arraycopy(models[i].valid, 0, valid, offset, models[i].n_classifiers);
                System.arraycopy(models[i].treeRuleSet, 0, treeRuleSet, offset, models[i].n_classifiers);
                offset += models[i].n_classifiers;   
            }
            
            
        }
    
}
