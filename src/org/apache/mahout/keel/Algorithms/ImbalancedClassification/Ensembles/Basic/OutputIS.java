/**
 * Created on 5-Feb-2005
 *
 * Clase implementada funciones para el manejo de ficheros de datos
 *
 */

/**
 * @author Salvador Garc�a L�pez
 *
 *
 */

package org.apache.mahout.keel.Algorithms.ImbalancedClassification.Ensembles.Basic;

import org.apache.mahout.keel.Dataset.*;
import org.core.*;

public class OutputIS {

  public static void escribeSalida (String nombreFichero, double instanciasIN[][], int instanciasOUT[], Attribute entradas[], Attribute salida, int nEntradas, String relation) {

    String cadena = "";
    int i, j;
    int aux;

    /*Printing input attributes*/
    cadena += "@relation "+ relation +"\n";
    for (i=0; i<nEntradas; i++) {
      cadena += "@attribute "+ entradas[i].getName()+" ";
      if (entradas[i].getType() == Attribute.NOMINAL) {
        cadena += "{";
        for (j=0; j<entradas[i].getNominalValuesList().size(); j++) {
          cadena += (String)entradas[i].getNominalValuesList().elementAt(j);
          if (j < entradas[i].getNominalValuesList().size() -1) {
            cadena += ", ";
          }
        }
        cadena += "}\n";
      } else {
        if (entradas[i].getType() == Attribute.INTEGER) {
          cadena += "integer";
          cadena += " ["+String.valueOf((int)entradas[i].getMinAttribute()) + ", " +  String.valueOf((int)entradas[i].getMaxAttribute())+"]\n";
        } else {
          cadena += "real";
          cadena += " ["+String.valueOf(entradas[i].getMinAttribute()) + ", " +  String.valueOf(entradas[i].getMaxAttribute())+"]\n";
        }
      }
    }

    /*Printing output attribute*/
    cadena += "@attribute "+ salida.getName()+" ";
    if (salida.getType() == Attribute.NOMINAL) {
      cadena += "{";
      for (j=0; j<salida.getNominalValuesList().size(); j++) {
        cadena += (String)salida.getNominalValuesList().elementAt(j);
        if (j < salida.getNominalValuesList().size() -1) {
          cadena += ", ";
        }
      }
      cadena += "}\n";
    } else {
      cadena += "integer ["+String.valueOf((int)salida.getMinAttribute()) + ", " + String.valueOf((int)salida.getMaxAttribute())+"]\n";
    }

    /*Printing the data*/
    cadena += "@data\n";

    Fichero.escribeFichero (nombreFichero, cadena);
    for (i=0; i<instanciasIN.length; i++) {
      cadena = "";
      for (j=0; j<instanciasIN[i].length; j++) {
        if (entradas[j].getType() == Attribute.REAL) {
          String number = String.valueOf((instanciasIN[i][j]*(entradas[j].getMaxAttribute()-entradas[j].getMinAttribute())) + entradas[j].getMinAttribute());
          String comma[] = number.split("\\.");
		  if (comma.length > 1) {
            int pos1, pos2;
			if (comma[1].indexOf("E") < 0) {
              pos1 = comma[1].indexOf("0000");
              pos2 = comma[1].indexOf("9999");
              if (pos1 >= 0) {
                comma[1] = comma[1].substring(0,pos1);
               if (comma[1].length() == 0) comma[1] = "0";
              } else if (pos2 >= 0) {
                comma[1] = comma[1].substring(0,pos2);
//              System.out.println(comma[1]);
                if (comma[1].length() == 0) {
                  comma[1] = "0";
                  int redondo = Integer.parseInt(comma[0]);
                  redondo++;
                  comma[0] = String.valueOf(redondo);
                } else {
                  long redondo = Long.parseLong(comma[1].substring(comma[1].length()-1));
                  redondo++;
                  comma[1] = comma[1].substring(0,comma[1].length()-1) + String.valueOf(redondo);
                }
			  }
            }
            cadena +=  comma[0] + "." + comma[1] + ",";
		  } else {
            cadena +=  comma[0] + ",";
		  }
        } else if (Attributes.getInputAttribute(j).getType() == Attribute.INTEGER) {
          cadena += String.valueOf((int)((instanciasIN[i][j]*(entradas[j].getMaxAttribute()-entradas[j].getMinAttribute())) + entradas[j].getMinAttribute())) + ",";
        } else {
          aux = (int) (instanciasIN[i][j] * (entradas[j].getNominalValuesList().size() - 1));
          cadena += (String)entradas[j].getNominalValuesList().elementAt(aux) + ",";
        }
      }
      if (salida.getType() == Attribute.INTEGER) {
        cadena += String.valueOf(instanciasOUT[i]);
      } else {
        cadena += (String)salida.getNominalValuesList().elementAt(instanciasOUT[i]);
      }

      cadena += "\n";
      Fichero.AnadirtoFichero (nombreFichero, cadena);
    }
  }

  public static void escribeSalida (String nombreFichero, InstanceSet datos, Attribute entradas[], Attribute salida, int nEntradas, String relation) {
    String cadena = "";
    int i, j;

    /*Printing input attributes*/
    cadena += "@relation "+ relation +"\n";
    for (i=0; i<nEntradas; i++) {
      cadena += "@attribute "+ entradas[i].getName()+" ";
      if (entradas[i].getType() == Attribute.NOMINAL) {
        cadena += "{";
        for (j=0; j<entradas[i].getNominalValuesList().size(); j++) {
          cadena += (String)entradas[i].getNominalValuesList().elementAt(j);
          if (j < entradas[i].getNominalValuesList().size() -1) {
            cadena += ", ";
          }
        }
        cadena += "}\n";
      } else {
        if (entradas[i].getType() == Attribute.INTEGER) {
          cadena += "integer";
          cadena += " ["+String.valueOf((int)entradas[i].getMinAttribute()) + ", " +  String.valueOf((int)entradas[i].getMaxAttribute())+"]\n";
        } else {
          cadena += "real";
          cadena += " ["+String.valueOf(entradas[i].getMinAttribute()) + ", " +  String.valueOf(entradas[i].getMaxAttribute())+"]\n";
        }
      }
    }

    /*Printing output attribute*/
    cadena += "@attribute "+ salida.getName()+" ";
    if (salida.getType() == Attribute.NOMINAL) {
      cadena += "{";
      for (j=0; j<salida.getNominalValuesList().size(); j++) {
        cadena += (String)salida.getNominalValuesList().elementAt(j);
        if (j < salida.getNominalValuesList().size() -1) {
          cadena += ", ";
        }
      }
      cadena += "}\n";
    } else {
      cadena += "integer ["+String.valueOf((int)salida.getMinAttribute()) + ", " + String.valueOf((int)salida.getMaxAttribute())+"]\n";
    }

    /*Printing the data*/
    cadena += "@data\n";

    Fichero.escribeFichero (nombreFichero, cadena);
    for (i=0; i<datos.getNumInstances(); i++) {
      cadena = "";
      cadena = datos.getInstance(i).toString();
      cadena += "\n";
      String bueno = "";
      for (j=0; j<cadena.length(); j++) {
        if (cadena.charAt(j)=='n' && (j+1)<cadena.length() && cadena.charAt(j+1)=='u' && (j+2)<cadena.length() && cadena.charAt(j+2)=='l' && (j+3)<cadena.length() && cadena.charAt(j+3) =='l') {
          bueno = bueno.concat("?");
          j+=3;
        } else {
          bueno = bueno.concat(String.valueOf(cadena.charAt(j)));
        }
      }
      Fichero.AnadirtoFichero (nombreFichero, bueno);
    }

  }


  public static void escribeSalidaNumber (String nombreFichero, double instanciasIN[][], double instanciasOUT[][], Attribute entradas[], Attribute salidas[], int nEntradas, int nSalidas, String relation) {

    String cadena = "";
    int i, j;

    /*Printing input attributes*/
    cadena += "@relation " + relation + "\n";
    for (i = 0; i < nEntradas; i++) {
      cadena += "@attribute " + entradas[i].getName() + " ";
      if (entradas[i].getType() == Attribute.NOMINAL) {
        cadena += "integer";
        cadena += " [" + String.valueOf(0) + ", " +
            String.valueOf(entradas[i].getNominalValuesList().size() - 1) +
            "]\n";
      }
      else {
        if (entradas[i].getType() == Attribute.INTEGER) {
          cadena += "integer";
          cadena += " [" + String.valueOf( (int) entradas[i].getMinAttribute()) +
              ", " + String.valueOf( (int) entradas[i].getMaxAttribute()) +
              "]\n";
        }
        else {
          cadena += "real";
          cadena += " [" + String.valueOf(entradas[i].getMinAttribute()) + ", " +
              String.valueOf(entradas[i].getMaxAttribute()) + "]\n";
        }
      }
    }

    /*Printing output attributes*/
    for (i = 0; i < nSalidas; i++) {
      cadena += "@attribute " + salidas[i].getName() + " ";
      if (salidas[i].getType() == Attribute.NOMINAL) {
        cadena += "integer";
        cadena += " [" + String.valueOf(0) + ", " +
            String.valueOf(salidas[i].getNominalValuesList().size() - 1) +
            "]\n";
      }
      else {
        if (salidas[i].getType() == Attribute.INTEGER) {
          cadena += "integer";
          cadena += " [" + String.valueOf( (int) salidas[i].getMinAttribute()) +
              ", " + String.valueOf( (int) salidas[i].getMaxAttribute()) +
              "]\n";
        }
        else {
          cadena += "real";
          cadena += " [" + String.valueOf(salidas[i].getMinAttribute()) + ", " +
              String.valueOf(salidas[i].getMaxAttribute()) + "]\n";
        }
      }
    }

    cadena += "@inputs ";
    for (i = 0; i < nEntradas - 1; i++) {
      cadena += entradas[i].getName() + ", ";
    }
    if (nEntradas > 0)
      cadena += entradas[nEntradas - 1].getName() + "\n";

    cadena += "@outputs ";
    for (i = 0; i < nSalidas - 1; i++) {
      cadena += salidas[i].getName() + ", ";
    }
    if (nSalidas > 0)
      cadena += salidas[nSalidas - 1].getName() + "\n";

      /*Printing the data*/
    cadena += "@data\n";

    Fichero.escribeFichero(nombreFichero, cadena);
    for (i = 0; i < instanciasIN.length; i++) {
      cadena = "";
      for (j = 0; j < instanciasIN[i].length; j++) {
        if (instanciasIN[i][j] == Double.POSITIVE_INFINITY) {
          cadena += "?,";
        } else {
          if (entradas[j].getType() == Attribute.REAL) {
            cadena += String.valueOf(instanciasIN[i][j]) + ",";
          }
          else {
            cadena += String.valueOf( (int) (instanciasIN[i][j])) + ",";
          }
        }
      }
      for (j = 0; j < instanciasOUT[i].length-1; j++) {
        if (salidas[j].getType() == Attribute.REAL) {
          cadena += String.valueOf(instanciasOUT[i][j]) + ",";
        }
        else {
          cadena += String.valueOf( (int) (instanciasOUT[i][j])) + ",";
        }
      }
      if (salidas[j].getType() == Attribute.REAL) {
        cadena += String.valueOf(instanciasOUT[i][j]) + "\n";
      }
      else {
        cadena += String.valueOf( (int) (instanciasOUT[i][j])) + "\n";
      }
      Fichero.AnadirtoFichero(nombreFichero, cadena);
    }
  }

  public static void escribeSalidaSinNull (String nombreFichero, InstanceSet datos) {

    String cadena = "";
    int i;

    cadena = datos.getHeader();
    cadena += "@data\n";

    Fichero.escribeFichero(nombreFichero, cadena);
    for (i=0; i<datos.getNumInstances(); i++) {
      cadena = "";
      cadena = datos.getInstance(i).toString();
      cadena += "\n";
      if (cadena.indexOf("null") == -1)
        Fichero.AnadirtoFichero (nombreFichero, cadena);
    }
  }

  public static void escribeSalidaRanging (String nombreFichero, double instanciasIN[][], double instanciasOUT[][], Attribute entradas[], Attribute salidas[], int nEntradas, int nSalidas, String relation) {

  String cadena = "";
  int i, j;

  /*Printing input attributes*/
  cadena += "@relation " + relation + "\n";
  for (i = 0; i < nEntradas; i++) {
    cadena += "@attribute " + entradas[i].getName() + " ";
    cadena += "real [0.0, 1.0]\n";
  }

  /*Printing output attributes*/
  for (i = 0; i < nSalidas; i++) {
    cadena += "@attribute " + salidas[i].getName() + " ";
    if (salidas[i].getType() == Attribute.NOMINAL) {
      cadena += "{";
      for (j=0; j<salidas[i].getNominalValuesList().size(); j++) {
        cadena += (String)salidas[i].getNominalValuesList().elementAt(j);
        if (j < salidas[i].getNominalValuesList().size() -1) {
          cadena += ", ";
        }
      }
      cadena += "}\n";
    }
    else {
      if (salidas[i].getType() == Attribute.INTEGER) {
        cadena += "integer";
        cadena += " [" + String.valueOf( (int) salidas[i].getMinAttribute()) +
            ", " + String.valueOf( (int) salidas[i].getMaxAttribute()) +
            "]\n";
      }
      else {
        cadena += "real";
        cadena += " [" + String.valueOf(salidas[i].getMinAttribute()) + ", " +
            String.valueOf(salidas[i].getMaxAttribute()) + "]\n";
      }
    }
  }

  cadena += "@inputs ";
  for (i = 0; i < nEntradas - 1; i++) {
    cadena += entradas[i].getName() + ", ";
  }
  if (nEntradas > 0)
    cadena += entradas[nEntradas - 1].getName() + "\n";

  cadena += "@outputs ";
  for (i = 0; i < nSalidas - 1; i++) {
    cadena += salidas[i].getName() + ", ";
  }
  if (nSalidas > 0)
    cadena += salidas[nSalidas - 1].getName() + "\n";

    /*Printing the data*/
  cadena += "@data\n";

  Fichero.escribeFichero(nombreFichero, cadena);
  for (i = 0; i < instanciasIN.length; i++) {
    cadena = "";
    for (j = 0; j < instanciasIN[i].length; j++) {
      if (instanciasIN[i][j] == Double.POSITIVE_INFINITY) {
        cadena += "?,";
      } else {
        if (entradas[j].getType() == Attribute.REAL) {
          cadena += String.valueOf(instanciasIN[i][j]) + ",";
        }
        else {
          cadena += String.valueOf(instanciasIN[i][j]) + ",";
        }
      }
    }
    for (j = 0; j < instanciasOUT[i].length-1; j++) {
      if (salidas[j].getType() == Attribute.REAL) {
        cadena += String.valueOf(instanciasOUT[i][j]) + ",";
      } else if (salidas[j].getType() == Attribute.INTEGER) {
        cadena += String.valueOf(instanciasOUT[i][j]) + ",";
      } else {
        cadena += (String)salidas[j].getNominalValuesList().elementAt((int)instanciasOUT[i][j]) + ",";
	  }
    }
    if (salidas[j].getType() == Attribute.REAL) {
      cadena += String.valueOf(instanciasOUT[i][j]) + "\n";
    } else if (salidas[j].getType() == Attribute.INTEGER) {
        cadena += String.valueOf(instanciasOUT[i][j]) + "\n";
    } else {
        cadena += (String)salidas[j].getNominalValuesList().elementAt((int)instanciasOUT[i][j]) + "\n";
    }
    Fichero.AnadirtoFichero(nombreFichero, cadena);
  }
}

  /*Required for HVDM distance*/
  public static void escribeSalida (String nombreFichero, double realIN[][], int nominalIN[][], boolean nulosIN[][], int instanciasOUT[], Attribute entradas[], Attribute salida, int nEntradas, String relation) {

		  String cadena = "";
		  int i, j;

		  /*Printing input attributes*/
		  cadena += "@relation "+ relation +"\n";
		  for (i=0; i<nEntradas; i++) {
			  cadena += "@attribute "+ entradas[i].getName()+" ";
			  if (entradas[i].getType() == Attribute.NOMINAL) {
				  cadena += "{";
				  for (j=0; j<entradas[i].getNominalValuesList().size(); j++) {
					  cadena += (String)entradas[i].getNominalValuesList().elementAt(j);
					  if (j < entradas[i].getNominalValuesList().size() -1) {
						  cadena += ", ";
					  }
				  }
				  cadena += "}\n";
			  } else {
				  if (entradas[i].getType() == Attribute.INTEGER) {
					  cadena += "integer";
					  cadena += " ["+String.valueOf((int)entradas[i].getMinAttribute()) + ", " +  String.valueOf((int)entradas[i].getMaxAttribute())+"]\n";
				  } else {
					  cadena += "real";
					  cadena += " ["+String.valueOf(entradas[i].getMinAttribute()) + ", " +  String.valueOf(entradas[i].getMaxAttribute())+"]\n";
				  }
			  }
		  }

		  /*Printing output attribute*/
		  cadena += "@attribute "+ salida.getName()+" ";
		  if (salida.getType() == Attribute.NOMINAL) {
			  cadena += "{";
			  for (j=0; j<salida.getNominalValuesList().size(); j++) {
				  cadena += (String)salida.getNominalValuesList().elementAt(j);
				  if (j < salida.getNominalValuesList().size() -1) {
					  cadena += ",";
				  }
			  }
			  cadena += "}\n";
		  } else {
			  cadena += "integer ["+String.valueOf((int)salida.getMinAttribute()) + ", " + String.valueOf((int)salida.getMaxAttribute())+"]\n";
		  }

		  /*Printing the data*/
		  cadena += "@data\n";

		  Fichero.escribeFichero (nombreFichero, cadena);
                  StringBuilder sb = new StringBuilder("");
                  //String cadenaTotal = "";
		  for (i=0; i<realIN.length; i++) {
			  cadena = "";
			  for (j=0; j<realIN[i].length; j++) {
				  if (nulosIN[i][j] == false) {
					  if (entradas[j].getType() == Attribute.REAL) {
						  cadena += String.valueOf(realIN[i][j]) + ",";
					  } else if (Attributes.getInputAttribute(j).getType() == Attribute.INTEGER) {
						  cadena += String.valueOf((int)(realIN[i][j]))+ ",";
					  } else {
						  cadena += (String)entradas[j].getNominalValuesList().elementAt(nominalIN[i][j]) + ",";
					  }
				  } else {				  
					  cadena += "?,";
				  }
			  }
			  if (salida.getType() == Attribute.INTEGER) {
				  cadena += String.valueOf(instanciasOUT[i]);
			  } else {
				  cadena += (String)salida.getNominalValuesList().elementAt(instanciasOUT[i]);
			  }
			  cadena += "\n";
                          sb.append(cadena);
		  }
		  
                            Fichero.AnadirtoFichero (nombreFichero, sb.toString());
	}//end-method
}

