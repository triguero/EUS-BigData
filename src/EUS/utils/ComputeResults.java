package EUS.utils;

public class ComputeResults {
	
	public static double computeAuc(int [][] matrix){
	    int [] classesDistribution = new int [matrix.length-1];  
		for(int i=0; i< matrix.length-1; i++){
	      for(int j=0; j< matrix[i].length-1; j++){	          	          
		    classesDistribution[i]+=matrix[i][j];	
		  } 	        	   
		}    
		int posClassId = 0;
		int posNumInstances = classesDistribution[0]; 
		for (int k=1; k<matrix.length-1; k++) {
		  if (classesDistribution[k] < posNumInstances) {
		    posClassId = k;
		 	posNumInstances = classesDistribution[k];
		   }
		}
		double tp_rate = 0.0;
		double fp_rate = 0.0;
		if(posClassId == 0){
		  tp_rate=((double)matrix[0][0]/(matrix[0][0]+matrix[0][1]));	
		  fp_rate=((double)matrix[1][0]/(matrix[1][0]+matrix[1][1]));
		}
		else{
		  fp_rate=((double)matrix[0][1]/(matrix[0][1]+matrix[0][0]));	
		  tp_rate=((double)matrix[1][1]/(matrix[1][1]+matrix[1][0]));	
		}
		return ((1+tp_rate-fp_rate)/2);
	  }

			  
	  public static double computeGM(int [][] matrix){
	    int [] classesDistribution = new int [matrix.length-1];  
		for(int i=0; i< matrix.length-1; i++){
		  for(int j=0; j< matrix[i].length-1; j++){	          	          
		    classesDistribution[i]+=matrix[i][j];	
		  } 	        	   
		}    
		int posClassId = 0;
		int posNumInstances = classesDistribution[0]; 
		for (int k=1; k<matrix.length-1; k++) {
		  if (classesDistribution[k] < posNumInstances) {
		    posClassId = k;
			posNumInstances = classesDistribution[k];
		  }
		}
		double sensisivity = 0.0;
		double specificity = 0.0;
		if(posClassId == 0){
		  sensisivity=((double)matrix[0][0]/(matrix[0][0]+matrix[0][1]));	
		  specificity=((double)matrix[1][1]/(matrix[1][1]+matrix[1][0]));
		}
		else{
	      specificity=((double)matrix[0][0]/(matrix[0][0]+matrix[0][1]));	
		  sensisivity=((double)matrix[1][1]/(matrix[1][1]+matrix[1][0]));	
	    }
		return (Math.sqrt(sensisivity*specificity));  
	  }
}
