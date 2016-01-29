/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010

	F. Herrera (herrera@decsai.ugr.es)
    L. Sánchez (luciano@uniovi.es)
    J. Alcalá-Fdez (jalcala@decsai.ugr.es)
    S. García (sglopez@ujaen.es)
    A. Fernández (alberto.fernandez@ujaen.es)
    J. Luengo (julianlm@decsai.ugr.es)

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see http://www.gnu.org/licenses/

 **********************************************************************/

package org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles;
import java.io.IOException;
import org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles.C45.C45;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * <p>Title: Main Class of the Program</p>
 *
 * <p>Description: It reads the configuration file (data-set files and parameters) and launch the algorithm</p>
 *
 * <p>Company: KEEL</p>
 *
 * @author Written by Alberto Fernandez (University of Jaen) 09/10/2012
 * @version 1.0
 */
public class Main {

	private parseParameters parameters;

	/** Default Constructor */
	public Main() {
	}

	/**
	 * It launches the algorithm
	 * @param confFile String it is the filename of the configuration file.
	 */
	private void execute(String confFile) {
		parameters = new parseParameters();
                
                String param = "seed = 48127491\n" +
                                "pruned = TRUE\n" +
                                "confidence = 0.25\n" +
                                "instancesPerLeaf = 2\n" +
                                "nClassifiers = 10\n" +
                                "ensembleType = ERUSBOOST\n" +
                                "train method = NORESAMPLING\n" +
                                "RUSBoost N prctg de la maj/ Quantity of balancing in SMOTE = 50\n" +
                                "ISmethod = QstatEUB_M_GM\n" +
                                "number of Bags for hybrid = 4\n";
                
		parameters.parseConfigurationString(param);

	    myDataset myDS = new myDataset();
	    try {
			System.out.println("\nReading the training set: " + "./data/ecoli-0_vs_1/ecoli-0_vs_1-5-1tra.dat");
			myDS.readClassificationSet("./data/ecoli-0_vs_1/ecoli-0_vs_1-5-1tra.dat", true);
		}
		catch (IOException e) {
			System.err.println("There was a problem while reading the input data-sets: " + e);
		}
                
             
		multi_C45 method = new multi_C45(myDS.getIS(), parameters);
		method.execute(true);

	}

	/**
	 * Main Program
	 * @param args It contains the name of the configuration file<br/>
	 * Format:<br/>
	 * <em>algorith = &lt;algorithm name></em><br/>
	 * <em>inputData = "&lt;training file&gt;" "&lt;validation file&gt;" "&lt;test file&gt;"</em> ...<br/>
	 * <em>outputData = "&lt;training file&gt;" "&lt;test file&gt;"</em> ...<br/>
	 * <br/>
	 * <em>seed = value</em> (if used)<br/>
	 * <em>&lt;Parameter1&gt; = &lt;value1&gt;</em><br/>
	 * <em>&lt;Parameter2&gt; = &lt;value2&gt;</em> ... <br/>
	 */
	public static void main(String args[]) {
		long t_ini = System.currentTimeMillis();
		Main program = new Main();
		System.out.println("Executing Algorithm.");
		program.execute(args[0]);
		long t_fin = System.currentTimeMillis();
		long t_exec = t_fin - t_ini;
		long hours = t_exec / 3600000;
		long rest = t_exec % 3600000;
		long minutes = rest / 60000;
		rest %= 60000;
		long seconds = rest / 1000;
		rest %= 1000;
		System.out.println("Execution Time: " + hours + ":" + minutes + ":" +
				seconds + "." + rest);
	}
}
