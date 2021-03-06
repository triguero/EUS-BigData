package org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import org.core.*;

import org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles.C45.C45;
import org.apache.mahout.keel.Dataset.InstanceSet;

import java.util.StringTokenizer;
import java.util.Vector;



/**
 * <p>Title: multi_C45</p>
 * <p>Description: Main class to compute the algorithm procedure
 * <p>Company: KEEL </p>
 * @author Mikel Galar Idoate (UPNA)
 * @author Modified by Alberto Fernandez (University of Jaen) 08/05/2014
 * @version 1.2
 * @since JDK1.6
 */
public class multi_C45 {

	parseParameters parameters;
	myDataset train, val, test;
	public static String outputTr, outputTst, ruleBaseFile;
	int instancesPerLeaf;
	public int n_classifiers;
	float confidence;
	boolean pruned;
	public boolean valid[];
	String fichTrain;
	
	public RuleBase[] treeRuleSet;           // Trees of the ensemble
	public myDataset actua_train_set;        // train data-set for the actual ensemble
	public Ensemble ensemble;                
	public String ensembleType; 
	public String evMeas;

	public Model_MapReduce model = null;
	private boolean somethingWrong = false; //to check if everything is correct.

	String targets[];
	

	
	/**
	 * Default constructor
	 */
	public multi_C45() {
	}

	public multi_C45(Model_MapReduce mMapRed2){
        treeRuleSet = mMapRed2.treeRuleSet.clone();
        valid = mMapRed2.valid.clone();
        ensembleType = mMapRed2.ensembleType;
        n_classifiers =  mMapRed2.n_classifiers;
        ensemble=new Ensemble();
        ensemble.classifier = this;
        ensemble.nClassifier =  mMapRed2.n_classifiers;
        ensemble.ensembleType=mMapRed2.ensembleType;
        ensemble.alfa = mMapRed2.alfa.clone();
	
	}
	/**
	 * It reads the data from the input files (training, validation and test) and parse all the parameters
	 * from the parameters array.
	 * @param parameters parseParameters It contains the input files, output files and parameters
	 */
	public multi_C45(parseParameters parameters) {

		this.parameters = parameters;
		train = new myDataset();
		val = new myDataset();
		test = new myDataset();
		fichTrain = parameters.getTrainingInputFile();
		try {
			System.out.println("\nReading the training set: " +
					parameters.getTrainingInputFile());
			train.readClassificationSet(parameters.getTrainingInputFile(), true);
			System.out.println("\nReading the validation set: " +
					parameters.getValidationInputFile());
			val.readClassificationSet(parameters.getValidationInputFile(), false);
			System.out.println("\nReading the test set: " +
					parameters.getTestInputFile());
			test.readClassificationSet(parameters.getTestInputFile(), false);
		}
		catch (IOException e) {
			System.err.println(
					"There was a problem while reading the input data-sets: " +
							e);
			somethingWrong = true;
		}

		outputTr = parameters.getTrainingOutputFile();
		outputTst = parameters.getTestOutputFile();

		ruleBaseFile = parameters.getOutputFile(0);

		//Now we parse the parameters
		pruned = parameters.getParameter(1).equalsIgnoreCase("TRUE");
		confidence = Float.parseFloat(parameters.getParameter(2));
		instancesPerLeaf = Integer.parseInt(parameters.getParameter(3));
		n_classifiers = Integer.parseInt(parameters.getParameter(4));
		ensembleType = parameters.getParameter(5);


		/* Create the ensemble! */   
		ensemble = new Ensemble(ensembleType, train, n_classifiers, this);

	}
        
    
    


    public multi_C45(InstanceSet IS, parseParameters parameters) {
    	this.parameters = parameters;
		train = new myDataset();
		try {
			//System.out.println("\nReading the training set: " +	parameters.getTrainingInputFile());
			train.readInstanceSet(IS);
	    	//this.context.progress();

		}
		catch (IOException e) {
			System.err.println(
					"There was a problem while reading the input data-sets: " +
							e);
			somethingWrong = true;
		}

		//Now we parse the parameters
		pruned = parameters.getParameter(1).equalsIgnoreCase("TRUE");
		confidence = Float.parseFloat(parameters.getParameter(2));
		instancesPerLeaf = Integer.parseInt(parameters.getParameter(3));
		n_classifiers = Integer.parseInt(parameters.getParameter(4));
		ensembleType = parameters.getParameter(5);

		//int minorityWindows = Integer.parseInt(parameters.getParameter(10));
		//int majorityWindows = Integer.parseInt(parameters.getParameter(11));


		/* Create the ensemble! */   
		ensemble = new Ensemble(ensembleType, train, n_classifiers, this); //,minorityWindows, majorityWindows
		
	   	//this.context.progress();

		
		targets=train.getOutputValues().clone();
		
		System.out.println("Possible classes - Ensemble : ");
		for(int i=0; i<targets.length; i++){
			  System.out.println(targets[i]);

		}
	}
    
    

	public String[] getTargets(){
    	return targets;
    }
	/**
	 * It launches the algorithm
	 */
	public void execute() {
		if (somethingWrong) { //We do not execute the program
			System.err.println("An error was found, the data-set has missing values.");
			System.err.println("Aborting the program");
			//We should not use the statement: System.exit(-1);
		}
		else {
		
			n_classifiers = ensemble.nClassifier;
			valid = new boolean[n_classifiers];
			treeRuleSet = new RuleBase[n_classifiers];

			
            System.out.println("Vamos con el execute!");

            
			/** While the algorithm has not end, and the number of classifier constructed is not reached... 
			 * we construct a new classifier for the ensemble
			 */
			boolean fin = false;
			for (int i = 0; i < n_classifiers && !fin; i++) {
				
	            System.out.println("Classifier: "+i);

				// we get the actual training data-set
				actua_train_set = ensemble.getDS();
				
	            System.out.println("Dataset obtained");

	            
                if (!ensembleType.equalsIgnoreCase("ERUSBOOST"))
                    actua_train_set.updateIS(train);

                System.out.println("llego al updateIS function!");
				// /* Databoost-IM has problems generating instances in Highly imbalanced data-sets */
				// if (actua_train_set.getnData() > 10000)
				// {
					// System.out.println("Databoost overflow!, nData = " + actua_train_set.getnData());
					// fin = true;
					// break;
				// }
				boolean mal = false;
				if (actua_train_set.getnClasses() == 2 && !actua_train_set.vacio())
				{
					// write the data-set which will be readed by C4.5 decision tree learning algorithm
//					Fichero.escribeFichero(outputTr +  "training.txt", actua_train_set.printDataSet());
					valid[i] = true;
					System.out.println("Training classifier[" + i + "]");
					// Construct the tree using the weights (they can be unirformly distributed)
					//C45 tree = new C45(outputTr +  "training.txt", pruned, confidence, instancesPerLeaf, ensemble.getWeights().clone());
                    C45 tree = new C45(actua_train_set.getIS(), pruned, confidence, instancesPerLeaf, ensemble.getWeights().clone());
					
        		

					try {
						tree.generateTree();
						

					}
					catch (Exception e) {
						System.err.println("Error!!");
						System.err.println(e.getMessage());
						System.exit( -1);
					}
					/* The tree is stored in a set of rules */
					//Fichero.escribeFichero("tree.txt", tree.printString());
					String cadenatree = tree.printString();
					obtainRules(cadenatree, i);
					
					

				   	
					if (treeRuleSet[i].size() == 0)
					{
						mal = true;
						int clase = tree.getPriorProbabilities()[0] > tree.getPriorProbabilities()[1] ? 0 : 1;
						// The a priori rule is introduced which predict the class with the greatest prior probability
						treeRuleSet[i].ruleBase.add(new Rule(train.getOutputValue(clase), actua_train_set));
					   	

					}

					treeRuleSet[i].coverExamples();

					treeRuleSet[i].coverExamples(ensemble.getWeights().clone());    //Step 2 
                    treeRuleSet[i].majClass = train.claseNumerica(train.claseMasFrecuente());
        			

				}
				else {
					System.out.println("Siempre aquí! "+i);
					valid[i] = false;
				}
				// Go to the next iteration of the ensemble!
				if (mal) {
					if ((!ensembleType.contains("EUNDERBAGGING")) && (ensemble.weightsBackup != null)) {
						ensemble.weights = ensemble.weightsBackup.clone();
						//i = i - 1;
					}
					else
						fin = ensemble.nextIteration();
					// ensemble.t = ensemble.t - 1;
				} else
					fin = ensemble.nextIteration();
				if (ensembleType.equalsIgnoreCase("EASYENSEMBLE") 
						|| ensembleType.equalsIgnoreCase("BALANCECASCADE"))
					i = ensemble.t - 1;
			}
                        
                        /*************/
                        /* MapReduce */
                        /*************/
                        
                        // Esto se har�a en el primer Map -> se genera el modelo y se escribe en el fichero (faltar�a poner el n�mero)
                        model = new Model_MapReduce(ensembleType, n_classifiers, valid, treeRuleSet, ensemble.alfa);                        
            			

/*
                        mMapRed.writeModel("/tmp/modelo-id.out");
                        
                        byte[] array = mMapRed.writeModel();
                        
                        // para comprobar que funciona...
                        treeRuleSet = null;
                        ensemble.alfa = null;
                        valid = null;
                        n_classifiers = 0;
                        mMapRed = null;
                        
                        // Aqu� se cargar�an todos los ficheros en un segundo Map (o reduce para juntarlos seg�n se prefiera)
                        // Como ejemplo r�pido cargo el mismo fichero varias veces
                        String[] files = {"/tmp/modelo-id.out", "/tmp/modelo-id.out", "/tmp/modelo-id.out", "/tmp/modelo-id.out","/tmp/modelo-id.out", "/tmp/modelo-id.out", "/tmp/modelo-id.out", "/tmp/modelo-id.out"};
                        Model_MapReduce mMapRed2 = new Model_MapReduce();
                        // MapRed2.readModels(files);
                       // mMapRed2.readModel(array);
                        mMapRed2.addModel(array);
                        mMapRed2.addModel(array);
                        mMapRed2.addModel(array);
                        mMapRed2.addModel(array);
                        mMapRed2.addModel(array);
                        mMapRed2.addModel(array);
                        
                        // Se juntan y se guardan en un fichero
                        mMapRed2.writeModel("/tmp/modelo-idJOIN.out");
                        
                        
                        
                        // Para comprobar que funcionar�a
                        treeRuleSet = mMapRed2.treeRuleSet.clone();
                        ensemble.alfa = mMapRed2.alfa.clone();
                        valid = mMapRed2.valid.clone();
                        ensembleType = mMapRed2.ensembleType;
                        n_classifiers =  mMapRed2.n_classifiers;
                          
                        // NOTA!!!: En el segundo map tambi�n habr�a que leer el fichero de configuraci�n (config.txt) y el dataset!
                        
                          
                        /*************/
                        /* MapReduce */
                        /*************/
                        
			//Finally we should fill the training and test output files
		//	double accTr = doOutput(this.train, this.outputTr);
//			double accTst = doOutput(this.test, this.outputTst);
//			writeOutput(accTr, accTst, this.ruleBaseFile);
//			ensemble.writeAUCError(this.outputTst);
		}
	}


	/**
	 * It generates the output file from a given dataset and stores it in a file
	 * @param dataset myDataset input dataset
	 * @param filename String the name of the file
	 * @return the Accuracy of the classifier
	 */
	private double doOutput(myDataset dataset, String filename) {
		double TP = 0, FP = 0, FN = 0, TN = 0;
		//String output = new String("");
		//output = dataset.copyHeader(); //we insert the header in the output file

		String outputTotal = dataset.copyHeader();
		String claseReal = "";
		String prediccion = "";
		String output2 = "";
		StringBuilder sb = new StringBuilder(dataset.getnData() * 5);
		int aciertos = 0;
		//We write the output for each example
		for (int i = 0; i < dataset.getnData(); i++) {
			claseReal = dataset.getOutputAsString(i);
			prediccion = this.classificationOutput(dataset.getExample(i));
			output2 = claseReal.concat(" ").concat(prediccion).concat("\n");
			// output += claseReal + " " + prediccion + "\n";
			if (claseReal.equalsIgnoreCase(prediccion)) {
				aciertos++;
			}

			if (claseReal.equalsIgnoreCase(prediccion) && claseReal.equalsIgnoreCase(train.claseMasFrecuente()))
				TN++;
			else if (claseReal.equalsIgnoreCase(prediccion) && !claseReal.equalsIgnoreCase(train.claseMasFrecuente()))
				TP++;
			else if (!claseReal.equalsIgnoreCase(prediccion) && claseReal.equalsIgnoreCase(train.claseMasFrecuente()))
				FP++;
			else
				FN++;

			sb.append(output2);
		}
		outputTotal += sb.toString();

		double TPrate = TP / (TP + FN);
		double TNrate = TN / (TN + FP);
		double gmean = Math.sqrt(TPrate * TNrate);
		double precision = TP / (TP + FP);
		double recall = TP / (TP + FN);
		double fmean = 2 * recall * precision / (1 * recall + precision);

		System.out.println("G-mean: " + gmean);
		System.out.println("F-mean: " + fmean);
		System.out.println("TPrate: " + TPrate);
		System.out.println("TNrate: " + TNrate);
		double FPrate = FP / (FP + TN);
		System.out.println("AUC: " + (1 + TPrate - FPrate) / 2);
//		Fichero.escribeFichero(filename, outputTotal);
		return (1.0 * aciertos / dataset.size());
	}

	/**
	 * It carries out the classification of a given dataset throughout the learning stage of the ensemble
	 * @param dataset the instance set
	 * @return accuracy for the current ensemble
	 */
	public double classify(myDataset dataset) {
		int aciertos = 0;
		//We write the output for each example
		for (int i = 0; i < dataset.getnData(); i++) {
			String claseReal = dataset.getOutputAsString(i);
			String prediccion = this.classificationOutput(dataset.getExample(i));
			if (claseReal.equalsIgnoreCase(prediccion)) {
				aciertos++;
			}
		}
		return (1.0 * aciertos / dataset.size());
	}	
	
	/**
	 * It returns the algorithm classification output given an input example
	 * @param example double[] The input example
	 * @return String the output generated by the algorithm
	 */
	public String classificationOutput(double[] example) {
		/**
      Here we should include the algorithm directives to generate the
      classification output from the input example
		 */
		return ensemble.computeClassScores(example);
	}
	
	/**
	 * It returns the algorithm classification output given an input example
	 * @param example double[] The input example
	 * @return String the output generated by the algorithm
	 */
	public PredPair classificationOutput(double[] example, String[]classes) {
		/**
      Here we should include the algorithm directives to generate the
      classification output from the input example
		 */
		return ensemble.computeClassScores(example,classes);
	}
	

	/** It returns the class index of the prediction of an example in the i^{th} classifier
	 * 
	 * @param i the classifier to be used
	 * @param example the example to be classified
	 * @return the predicted class index
	 */
	protected int obtainClass(int i, double[] example)
	{
		if (valid[i]) {
			String clase = "?";
			for (int j = 0; (j < treeRuleSet[i].size()) && (clase.equals("?"));
					j++) {
				if (treeRuleSet[i].ruleBase.get(j).covers(example)) {
					clase = treeRuleSet[i].ruleBase.get(j).clase;
				}
			}
			//System.out.println("Clase Predicha: "+clase);
			int clase_num = train.claseNumerica(clase);
			if (clase_num == -1)
			{
				// System.out.println("No da la clase!!!!!");
				clase_num = train.claseNumerica(train.claseMasFrecuente());
			}
			return clase_num;
		}
		else{
			//System.out.println("Cagada obtain classe");
			return -1;
			
		}
	}
	
	protected int obtainClass(int i, double[] example, String classes[])
	{
		if (valid[i]) {
			String clase = "?";
			for (int j = 0; (j < treeRuleSet[i].size()) && (clase.equals("?"));
					j++) {
				if (treeRuleSet[i].ruleBase.get(j).covers(example)) {
					clase = treeRuleSet[i].ruleBase.get(j).clase;
				}
			}
		//	System.out.println("Clase Predicha: "+clase);
			int clase_num=-1; // = train.claseNumerica(clase);

			for(int j=0; j<classes.length;j++){
				if(clase.equalsIgnoreCase(classes[j])){
					clase_num=j;
				}
			}
			if (clase_num == -1)
			{
				// System.out.println("No da la clase!!!!!");
				//clase_num = train.claseNumerica(train.claseMasFrecuente());
				clase_num=treeRuleSet[i].majClass; //TODO
			}
			return clase_num;
		}
		else{
			//System.out.println("Cagada obtain classe");
			return -1;
			
		}
	}
	

	/** It obtains the confidence on the prediction of the example in the i^{th} classifier
	 * 
	 * @param i the classifier to be used
	 * @param example the example to be classified
	 * @return the confidence on the prediction
	 */
	
	protected double obtainConfidence(int i, double[] example )
	{
		double confianza = 0;

		if (valid[i]) {
			String clase = "?";
			for (int j = 0; (j < treeRuleSet[i].size()) && (clase.equals("?"));
					j++) {
				if (treeRuleSet[i].ruleBase.get(j).covers(example)) {
					clase = treeRuleSet[i].ruleBase.get(j).clase;
					double nCubiertosOK = treeRuleSet[i].ruleBase.get(j).fCubiertosOK; //.cubiertosOK();
					double nCubiertos = treeRuleSet[i].ruleBase.get(j).fCubiertos;//.cubiertos();
					if (nCubiertos == 0)
						confianza = 0;
					else
						confianza = (ensemble.nData * nCubiertosOK + 1) / (ensemble.nData * nCubiertos + 2);
				}
			}
			
			int clase_num= train.claseNumerica(clase);
			

			if (clase_num == -1)
				confianza = 0.5;
			return confianza;
		}
		else
		{
			return 0.5;
		}
	}

	
	protected double obtainConfidence(int i, double[] example , String [] classes)
	{
		double confianza = 0;

		if (valid[i]) {
			String clase = "?";
			for (int j = 0; (j < treeRuleSet[i].size()) && (clase.equals("?"));
					j++) {
				if (treeRuleSet[i].ruleBase.get(j).covers(example)) {
					clase = treeRuleSet[i].ruleBase.get(j).clase;
					double nCubiertosOK = treeRuleSet[i].ruleBase.get(j).fCubiertosOK; //.cubiertosOK();
					double nCubiertos = treeRuleSet[i].ruleBase.get(j).fCubiertos;//.cubiertos();
					if (nCubiertos == 0)
						confianza = 0;
					else
						confianza = (ensemble.nData * nCubiertosOK + 1) / (ensemble.nData * nCubiertos + 2);
				}
			}
			
			int clase_num=-1; // = train.claseNumerica(clase);

			for(int j=0; j<classes.length;j++){
				if(clase.equalsIgnoreCase(classes[j])){
					clase_num=j;
				}
			}
			

			if (clase_num == -1)
				confianza = 0.5;
			return confianza;
		}
		else
		{
			return 0.5;
		}
	}

	/**
	 * It extracts the rule set from a given file exported by the C4.5 classifier
	 * @param treeString the contain of the file (rule set)
	 * @param classifier classifier id of the ensemble
	 */
	private void obtainRules(String treeString, int classifier) {
		String rules = new String("");
		StringTokenizer lines = new StringTokenizer(treeString, "\n"); //read lines
		String line = lines.nextToken(); //First line @TotalNumberOfNodes X
		line = lines.nextToken(); //Second line @NumberOfLeafs Y
		//The tree starts
		Vector <String>variables = new Vector<String>();
		Vector <String>values = new Vector<String>();
		Vector <String>operators = new Vector<String>();
		int contador = 0;
		while (lines.hasMoreTokens()) {
			line = lines.nextToken();
			StringTokenizer field = new StringTokenizer(line, " \t");
			String cosa = field.nextToken(); //Possibilities: "if", "elseif", "class"
			if (cosa.compareToIgnoreCase("if") == 0) {
				field.nextToken(); //(
				variables.add(field.nextToken()); //variable name (AttX, X == position)
				operators.add(field.nextToken()); //One of three: "=", "<=", ">"
				values.add(field.nextToken()); //Value
			}
			else if (cosa.compareToIgnoreCase("elseif") == 0) {
				int dejar = Integer.parseInt(field.nextToken());
				for (int i = variables.size() - 1; i >= dejar; i--) {
					variables.remove(variables.size() - 1);
					operators.remove(operators.size() - 1);
					values.remove(values.size() - 1);
				}
				field.nextToken(); //(
				variables.add(field.nextToken()); //variable name (AttX, X == position)
				operators.add(field.nextToken()); //One of three: "=", "<=", ">"
				values.add(field.nextToken()); //Value
			}
			else { //Class --> rule generation
				field.nextToken(); // =
				contador++; //I have a new rule
				rules += "\nRULE-" + contador + ": IF ";
				int i;
				for (i = 0; i < variables.size() - 1; i++) {
					rules += (String) variables.get(i) + " " + (String) operators.get(i) +
							" " + (String) values.get(i) + " AND ";
				}
				rules += (String) variables.get(i) + " " + (String) operators.get(i) +
						" " + (String) values.get(i);
				rules += " THEN class = " + field.nextToken();
				variables.remove(variables.size() - 1);
				operators.remove(operators.size() - 1);
				values.remove(values.size() - 1);
			}
		}
		treeRuleSet[classifier] = new RuleBase(actua_train_set, rules);
	}

	/**
	 * It writes on a file the full ensemble (C4.5 rule sets)
	 * @param accTr Training accuracy
	 * @param accTst Test accuracy
	 */	
	public void writeOutput(double accTr, double accTst, String ruleBaseFile) {
		for (int i = 0; i < treeRuleSet.length; i++) {
		}

		Fichero.escribeFichero(ruleBaseFile,"");
		for (int i = 0; i < ensemble.nClassifier; i++) {
			if (valid[i]) {
				Files.addToFile(ruleBaseFile, "@Classifier number " + i + ": \n");
				Files.addToFile(ruleBaseFile, treeRuleSet[i].printStringF() + "\n");
			}
			else {
				// System.out.println("Not valid!");
			}
		}
		Files.addToFile(ruleBaseFile, "Accuracy in training: " + accTr + "\n");
		Files.addToFile(ruleBaseFile, "Accuracy in test: " + accTst + "\n");

		System.out.println("Accuracy in training: " + accTr);
		System.out.println("Accuracy in test: " + accTst);
		System.out.println("Algorithm Finished");
	}


}
