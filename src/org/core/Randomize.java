/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010
	
	F. Herrera (herrera@decsai.ugr.es)
    L. Sanchez (luciano@uniovi.es)
    J. Alcala-Fdez (jalcala@decsai.ugr.es)
    S. Garcia (sglopez@ujaen.es)
    A. Fernandez (alberto.fernandez@ujaen.es)
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

package org.core;
import java.util.*;

public class Randomize {
	private static long Seed;
	private static MTwister generador = new MTwister();
	public static void setSeed (long semilla) {
		Seed = semilla;
		generador.init_genrand(Seed);
	}
	/** Rand computes a psuedo-random float value between 0 and 1, excluding 1 
	 * @return A uniform-distributed real value in [0,1) 
	 */
	public static double Rand () {
		return (generador.genrand_res53());
	}
	/** RandOpen computes a psuedo-random float value between 0 and 1, excluding 0 and 1  
	 * @return A uniform-distributed real value in (0,1)
	 */
	public static double RandOpen () {
		return (generador.genrand_real3());
	}
	/** RandClosed computes a psuedo-random float value between 0 and 1 inclusive  
	 * @return A uniform-distributed real value in [0,1]
	 */
	public static double RandClosed () {
		return (generador.genrand_real1());
	}
	/** RandGaussian generates a standardized gaussian random number  
	 * @return A normal-distributed real value with mean 0 and standard deviation equal to 1
	 */
	public static double RandGaussian () {
		return (generador.genrand_gaussian());
	}

	/** Randint gives an integer value between low and high, excluding high
	 * @param low Lower bound (included)
	 * @param high Upper bound (NOT included)
	 * @return A uniform-distributed integer value in [low,high)
	 */
	public static int Randint (int low, int high) {
		return ((int) (low + (high - low) * generador.genrand_res53()));
	}
	/** RandintOpen gives an integer value between low and high, excluding 0 and 1
	 * @param low Lower bound (NOT included)
	 * @param high Upper bound (NOT included)
	 * @return A uniform-distributed integer value in (low,high)
	 */
	public static int RandintOpen (int low, int high) {
		//we use low+1, to avoid that "low" could appear, since genrand_res53()
		//draws a real value in [0,1)
		return ((int) ((low+1) + (high - (low+1)) * generador.genrand_res53()));
	}
	/** RandintClosed gives an integer value between low and high inclusive
	 * @param low Lower bound (included)
	 * @param high Upper bound (included)
	 * @return A uniform-distributed integer value in [low,high]
	 */
	public static int RandintClosed (int low, int high) {
		//since genrand_res53() generates a double in [0,1), we increment
		//high by one, so "high" can appear with same probability as the rest of
		//numbers in the interval
		return ((int) (low + ((high+1) - low) * generador.genrand_res53()));
	}
	/** Randdouble gives an double value between low and high, excluding high
	 * @param low Lower bound (included)
	 * @param high Upper bound (NOT included)
	 * @return A uniform-distributed real value in [low,high)
	 */
	public static double Randdouble (double low, double high) {
		return (low + (high-low) * generador.genrand_res53());
	}
	/** RanddoubleOpen gives an double value between low and high, excluding low and high
	 * @param low Lower bound (NOT included)
	 * @param high Upper bound (NOT included)
	 * @return A uniform-distributed real value in (low,high)
	 */
	public static double RanddoubleOpen (double low, double high) {
		return (low + (high-low) * generador.genrand_real3());
	}
	/** RanddoubleClosed gives an double value between low and high inclusive
	 * @param low Lower bound (included)
	 * @param high Upper bound (included)
	 * @return A uniform-distributed real value in [low,high]
	 */
	public static double RanddoubleClosed (double low, double high) {
		return (low + (high-low) * generador.genrand_real1());
	}
}



