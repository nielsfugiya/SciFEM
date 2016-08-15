/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2007 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Moritz Allmaras, Texas A&M University, 2007
 */




#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/base/timer.h>

namespace Step29
{
  using namespace dealii;

  Tensor<1,2> wave_number_2d;



  // First we define a class for the function representing the Dirichlet
  // boundary values. This is borrowed from step-29
  //
  //In our project, we first implement a plane wave source, so it just returns cos(k*x)
  // for the real part $v$ and sin(k*x) for the imaginary part $w$ at the point where
  //it is evaluated.
  template <int dim>
  class DirichletBoundaryValues : public Function<dim>
  {
  public:
    DirichletBoundaryValues() : Function<dim> (2) {};

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &values) const;

    virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                    std::vector<Vector<double> >   &value_list) const;
  };


  template <int dim>
  inline
  void DirichletBoundaryValues<dim>::vector_value (const Point<dim> &p,
                                                   Vector<double>   &values) const
  {
    Assert (values.size() == 2, ExcDimensionMismatch (values.size(), 2));

    values(0) =  cos(wave_number_2d * p);
    values(1) =  -sin(wave_number_2d * p);
  }


  template <int dim>
  void DirichletBoundaryValues<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                                        std::vector<Vector<double> >   &value_list) const
  {
    Assert (value_list.size() == points.size(),
            ExcDimensionMismatch (value_list.size(), points.size()));

    for (unsigned int p=0; p<points.size(); ++p)
    {
      DirichletBoundaryValues<dim>::vector_value (points[p], value_list[p]);
    }
  }

  // @sect3{The <code>ParameterReader</code> class}

  // The next class is also borrowed from step29, but we have kept the usefule part only.
  // This class is responsible for preparing the ParameterHandler object
  // and reading parameters from an input file.  It includes a function
  // <code>declare_parameters</code> that declares all the necessary
  // parameters and a <code>read_parameters</code> function that is called
  // from outside to initiate the parameter reading process.
  class ParameterReader : public Subscriptor
  {
  public:
    ParameterReader(ParameterHandler &);
    void read_parameters(const std::string);

  private:
    void declare_parameters();
    ParameterHandler &prm;
  };

  // The constructor stores a reference to the ParameterHandler object that is
  // passed to it:
  ParameterReader::ParameterReader(ParameterHandler &paramhandler)
    :
    prm(paramhandler)
  {}

  // @sect4{<code>ParameterReader::declare_parameters</code>}

  // The <code>declare_parameters</code> function declares all the parameters
  // that our ParameterHandler object will be able to read from input files,
  // along with their types, range conditions and the subsections they appear
  // in. We will wrap all the entries that go into a section in a pair of
  // braces to force the editor to indent them by one level, making it simpler
  // to read which entries together form a section:
  void ParameterReader::declare_parameters()
  {
    prm.enter_subsection ("Mesh & geometry parameters");
    {
      prm.declare_entry("Number of refinements", "6",
                        Patterns::Integer(0),
                        "Number of global mesh refinement steps "
                        "applied to initial coarse grid");

      /*prm.declare_entry("Focal distance", "0.3",
                        Patterns::Double(0),
                        "Distance of the focal point of the lens "
                        "to the x-axis");*/
    }
    prm.leave_subsection ();

    // The next subsection is devoted to the physical parameters appearing in
    // the equation, which are the frequency $\omega$ and wave speed
    // $c$. Again, both need to lie in the half-open interval $[0,\infty)$
    // represented by calling the Patterns::Double class with only the left
    // end-point as argument:
    prm.enter_subsection ("Physical constants");
    {
      prm.declare_entry("c", "1.5e5",
                        Patterns::Double(0),
                        "Wave speed");

      prm.declare_entry("omega", "5.0e7",
                        Patterns::Double(0),
                        "Frequency");
    }
    prm.leave_subsection ();

    // Last but not least we would like to be able to change some properties
    // of the output, like filename and format, through entries in the
    // configuration file, which is the purpose of the last subsection:
    prm.enter_subsection ("Output parameters");
    {
      prm.declare_entry("Output file", "solution",
                        Patterns::Anything(),
                        "Name of the output file (without extension)");

      // Since different output formats may require different parameters for
      // generating output (like for example, postscript output needs
      // viewpoint angles, line widths, colors etc), it would be cumbersome if
      // we had to declare all these parameters by hand for every possible
      // output format supported in the library. Instead, each output format
      // has a <code>FormatFlags::declare_parameters</code> function, which
      // declares all the parameters specific to that format in an own
      // subsection. The following call of
      // DataOutInterface<1>::declare_parameters executes
      // <code>declare_parameters</code> for all available output formats, so
      // that for each format an own subsection will be created with
      // parameters declared for that particular output format. (The actual
      // value of the template parameter in the call, <code>@<1@></code>
      // above, does not matter here: the function does the same work
      // independent of the dimension, but happens to be in a
      // template-parameter-dependent class.)  To find out what parameters
      // there are for which output format, you can either consult the
      // documentation of the DataOutBase class, or simply run this program
      // without a parameter file present. It will then create a file with all
      // declared parameters set to their default values, which can
      // conveniently serve as a starting point for setting the parameters to
      // the values you desire.
      DataOutInterface<1>::declare_parameters (prm);
    }
    prm.leave_subsection ();
  }


  // @sect4{<code>ParameterReader::read_parameters</code>}

  // This is the main function in the ParameterReader class.  It gets called
  // from outside, first declares all the parameters, and then reads them from
  // the input file whose filename is provided by the caller. After the call
  // to this function is complete, the <code>prm</code> object can be used to
  // retrieve the values of the parameters read in from the file:
  void ParameterReader::read_parameters (const std::string parameter_file)
  {
    declare_parameters();
    prm.read_input (parameter_file);
  }

  // @sect3{The <code>ComputeIntensity</code> class}

  // As mentioned in the introduction, the quantity that we are really after
  // is the spatial distribution of the intensity of the ultrasound wave,
  // which corresponds to $|u|=\sqrt{v^2+w^2}$. Now we could just be content
  // with having $v$ and $w$ in our output, and use a suitable visualization
  // or postprocessing tool to derive $|u|$ from the solution we
  // computed. However, there is also a way to output data derived from the
  // solution in deal.II, and we are going to make use of this mechanism here.

  // So far we have always used the DataOut::add_data_vector function to add
  // vectors containing output data to a DataOut object.  There is a special
  // version of this function that in addition to the data vector has an
  // additional argument of type DataPostprocessor. What happens when this
  // function is used for output is that at each point where output data is to
  // be generated, the DataPostprocessor::compute_derived_quantities_scalar or
  // DataPostprocessor::compute_derived_quantities_vector function of the
  // specified DataPostprocessor object is invoked to compute the output
  // quantities from the values, the gradients and the second derivatives of
  // the finite element function represented by the data vector (in the case
  // of face related data, normal vectors are available as well). Hence, this
  // allows us to output any quantity that can locally be derived from the
  // values of the solution and its derivatives.  Of course, the ultrasound
  // intensity $|u|$ is such a quantity and its computation doesn't even
  // involve any derivatives of $v$ or $w$.

  // In practice, the DataPostprocessor class only provides an interface to
  // this functionality, and we need to derive our own class from it in order
  // to implement the functions specified by the interface. In the most
  // general case one has to implement several member functions but if the
  // output quantity is a single scalar then some of this boilerplate code can
  // be handled by a more specialized class, DataPostprocessorScalar and we
  // can derive from that one instead. This is what the
  // <code>ComputeIntensity</code> class does:
  template <int dim>
  class ComputeIntensity : public DataPostprocessorScalar<dim>
  {
  public:
    ComputeIntensity ();

    virtual
    void
    compute_derived_quantities_vector (const std::vector< Vector< double > > &uh,
                                       const std::vector< std::vector< Tensor< 1, dim > > > &duh,
                                       const std::vector< std::vector< Tensor< 2, dim > > > &dduh,
                                       const std::vector< Point< dim > > &normals,
                                       const std::vector<Point<dim> > &evaluation_points,
                                       std::vector< Vector< double > > &computed_quantities) const;
  };

  // In the constructor, we need to call the constructor of the base class
  // with two arguments. The first denotes the name by which the single scalar
  // quantity computed by this class should be represented in output files. In
  // our case, the postprocessor has $|u|$ as output, so we use "Intensity".
  //
  // The second argument is a set of flags that indicate which data is needed
  // by the postprocessor in order to compute the output quantities.  This can
  // be any subset of update_values, update_gradients and update_hessians
  // (and, in the case of face data, also update_normal_vectors), which are
  // documented in UpdateFlags.  Of course, computation of the derivatives
  // requires additional resources, so only the flags for data that is really
  // needed should be given here, just as we do when we use FEValues objects.
  // In our case, only the function values of $v$ and $w$ are needed to
  // compute $|u|$, so we're good with the update_values flag.
  template <int dim>
  ComputeIntensity<dim>::ComputeIntensity ()
    :
    DataPostprocessorScalar<dim> ("Intensity",
                                  update_values)
  {}

  // The actual postprocessing happens in the following function.  Its inputs
  // are a vector representing values of the function (which is here
  // vector-valued) representing the data vector given to
  // DataOut::add_data_vector, evaluated at all evaluation points where we
  // generate output, and some tensor objects representing derivatives (that
  // we don't use here since $|u|$ is computed from just $v$ and $w$, and for
  // which we assign no name to the corresponding function argument).  The
  // derived quantities are returned in the <code>computed_quantities</code>
  // vector.  Remember that this function may only use data for which the
  // respective update flag is specified by
  // <code>get_needed_update_flags</code>. For example, we may not use the
  // derivatives here, since our implementation of
  // <code>get_needed_update_flags</code> requests that only function values
  // are provided.
  template <int dim>
  void
  ComputeIntensity<dim>::compute_derived_quantities_vector (
    const std::vector< Vector< double > >                  &uh,
    const std::vector< std::vector< Tensor< 1, dim > > >  & /*duh*/,
    const std::vector< std::vector< Tensor< 2, dim > > >  & /*dduh*/,
    const std::vector< Point< dim > >                     & /*normals*/,
    const std::vector<Point<dim> >                        & /*evaluation_points*/,
    std::vector< Vector< double > >                        &computed_quantities
  ) const
  {
    Assert(computed_quantities.size() == uh.size(),
           ExcDimensionMismatch (computed_quantities.size(), uh.size()));

    for (unsigned int i=0; i<computed_quantities.size(); i++)
      {
        Assert(computed_quantities[i].size() == 1,
               ExcDimensionMismatch (computed_quantities[i].size(), 1));
        Assert(uh[i].size() == 2, ExcDimensionMismatch (uh[i].size(), 2));

        computed_quantities[i](0) = std::sqrt(uh[i](0)*uh[i](0) + uh[i](1)*uh[i](1));
      }
  }

  // Besides the function that represents the exact solution, we also need a
  // function which we can use as right hand side when assembling the linear
  // system of discretized equations. This is accomplished using the following
  // class and the following definition of its function.

  //Note: We implement two kind of waves in our scattering system: plane wave and spherical wave.

   template<int dim>
   class RightHandSide : public Function<dim>
   {
   public:
     RightHandSide() : Function<dim>() {}
     virtual std::complex<double> PlaneWave(const Point<dim> &p, const unsigned int component = 0) const;
     virtual std::complex<double> SphericalWave(const Point<dim> &p, const unsigned int component = 0) const;
     virtual void value_list (const std::vector<Point<dim> > &points,
                              std::vector<std::complex<double>>            &values,
                              const unsigned int              component = 0) const;
   };

   // First, we implement a plane wave for right hand side
   template <int dim>
   std::complex<double> RightHandSide<dim>::PlaneWave(const Point<dim> &p, const unsigned int) const
   {

   	return std::complex<double> (cos(wave_number_2d * p), -sin(wave_number_2d * p));
   }
   // Second, we implement a spherical wave for right hand side
   template<int dim>
   std::complex<double> RightHandSide<dim>::SphericalWave(const Point<dim> &p, const unsigned int) const
   {

       if (p.norm()<0.5)
	   {
		   return std::complex<double> (sin(wave_number_2d.norm() * p.norm()), cos(wave_number_2d.norm() * p.norm())) * (/*wave_number_2d.norm()*/1 / p.norm());
	   }
	   else
	   {
		   return std::complex<double> (0,0);
	   }
   }
   template <int dim>
   void RightHandSide<dim>::value_list (const std::vector<Point<dim>> &points,
                                      std::vector<std::complex<double>>            &values,
                                      const unsigned int              component) const
   {
     const unsigned int n_points = points.size();

     Assert (values.size() == n_points,
             ExcDimensionMismatch (values.size(), n_points));

     Assert (component == 0,
             ExcIndexRange (component, 0, 1));

     for (unsigned int i=0; i<n_points; ++i)
     {
         //values[i] = RightHandSide::PlaneWave(points[i]);
         values[i] = RightHandSide::SphericalWave(points[i]);
     }

   }

  // In this class we implement the refractive index for the scatterer.
  // Laplacian * u + n^2 u = f. It difines the parameter n.
  template <int dim>
  class RefractiveIndex: public Function<dim>
  {
  public:
    RefractiveIndex () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<double>            &values,
                             const unsigned int              component = 0) const;
  };
  // Here we define a homogeneous scatterer with refractive index 4.
  // Other region between the scatterer and the PML, we let it be vacuum(n = 1).
  template <int dim>
  double RefractiveIndex<dim>::value (const Point<dim> &p,
                                  const unsigned int) const
  {
    if (p.square()<0.3*0.3)
      return 4.0;
    else
      return 1;
  }

  template <int dim>
  void RefractiveIndex<dim>::value_list (const std::vector<Point<dim> > &points,
                                     std::vector<double>            &values,
                                     const unsigned int              component) const
  {
    const unsigned int n_points = points.size();
    Assert (values.size() == n_points,
            ExcDimensionMismatch (values.size(), n_points));

    Assert (component == 0,
            ExcIndexRange (component, 0, 1));

    for (unsigned int i=0; i<n_points; ++i)
      {
          values[i] = RefractiveIndex::value(points[i]);
      }
  }

  //This is the core class of our project. In this class, we define all we need for a perfectly matched layer.
  //
  template<int dim>
  class PerfectlyMatchedLayer : public Function<dim>
  {
  public:
    PerfectlyMatchedLayer() : Function<dim>() {}
    //sigma function used for computing the coefficients in PML.
    //it is equivalent to a conductivity in physics.
    //
    virtual double sigma(double x) const;

    // this define the tensor coefficients in the PML region.
    //In this lambda00 is the coefficient mutiply the refractive index
    //lambda11, lambda22 and lambda33 are diagonal entries of the tensor coefficient mutiply the gradient of the
    //complex field. It is equiilant to anisotropic medium in physics.
    virtual std::complex<double> lambda00(const Point<dim> &p, const unsigned int component = 0) const;
    virtual std::complex<double> lambda11(const Point<dim> &p, const unsigned int component = 0) const;
    virtual std::complex<double> lambda22(const Point<dim> &p, const unsigned int component = 0) const;
    virtual std::complex<double> lambda33(const Point<dim> &p, const unsigned int component = 0) const;

    virtual void lambda00_list (const std::vector<Point<dim> > &points,
                             std::vector<std::complex<double>>            &values,
                             const unsigned int              component = 0) const;
    virtual void lambda11_list (const std::vector<Point<dim> > &points,
                             std::vector<std::complex<double>>            &values,
                             const unsigned int              component = 0) const;
    virtual void lambda22_list (const std::vector<Point<dim> > &points,
                             std::vector<std::complex<double>>            &values,
                             const unsigned int              component = 0) const;
    virtual void lambda33_list (const std::vector<Point<dim> > &points,
                             std::vector<std::complex<double>>            &values,
                             const unsigned int              component = 0) const;
  };

  template<int dim>
  std::complex<double> PerfectlyMatchedLayer<dim>::lambda00 (const Point<dim> &p,
                                     const unsigned int /*component*/) const
  {
	     std::complex<double> s1(1, sigma(p[0]));
	     std::complex<double> s2(1, sigma(p[1]));

		  switch (dim)
		  {
		  case 2:  return s1;//return s1 * s2;
		  case 3: {std::complex<double> s3(1, sigma(p[2])); return s1 * s2  * s3;}
		  default:
			  abort ();
		  }
  }

  template <int dim>
  void PerfectlyMatchedLayer<dim>::lambda00_list (const std::vector<Point<dim>> &points,
                                     std::vector<std::complex<double>>            &values,
                                     const unsigned int              component) const
  {
    const unsigned int n_points = points.size();

    Assert (values.size() == n_points,
            ExcDimensionMismatch (values.size(), n_points));

    Assert (component == 0,
            ExcIndexRange (component, 0, 1));

    for (unsigned int i=0; i<n_points; ++i)
      {
          values[i] = PerfectlyMatchedLayer::lambda00(points[i]);
      }
  }


  template <int dim>
  std::complex<double> PerfectlyMatchedLayer<dim>::lambda11 (const Point<dim> &p,
                                     const unsigned int /*component*/) const
  {
	     std::complex<double> s1(1, sigma(p[0]));
	     std::complex<double> s2(1, sigma(p[1]));
		  switch (dim)
		  {
		  case 2:  return std::complex<double> (1,0)/s1;//return s2 / s1;
		  case 3: {std::complex<double> s3(1, sigma(p[2])); return (s2 * s3) / s1;}
		  default:
			  abort ();
		  }
  }
  template <int dim>
  void PerfectlyMatchedLayer<dim>::lambda11_list (const std::vector<Point<dim>> &points,
                                     std::vector<std::complex<double>>            &values,
                                     const unsigned int              component) const
  {
    const unsigned int n_points = points.size();

    Assert (values.size() == n_points,
            ExcDimensionMismatch (values.size(), n_points));

    Assert (component == 0,
            ExcIndexRange (component, 0, 1));

    for (unsigned int i=0; i<n_points; ++i)
    {
        values[i] = PerfectlyMatchedLayer::lambda11(points[i]);
    }
  }

  template<int dim>
  std::complex<double> PerfectlyMatchedLayer<dim>::lambda22 (const Point<dim> &p,
                                     const unsigned int /*component*/) const
  {
	     std::complex<double> s1(1.0, sigma(p[0]));
	     std::complex<double> s2(1.0, sigma(p[1]));
		  switch (dim)
		  {
		  case 2: return s1; //return s1 / s2;
		  case 3: {std::complex<double> s3(1, sigma(p[2])); return (s1 * s3) / s2;}
		  default:
			  abort ();
		  }
  }
  template <int dim>
  void PerfectlyMatchedLayer<dim>::lambda22_list (const std::vector<Point<dim>> &points,
                                     std::vector<std::complex<double>>            &values,
                                     const unsigned int              component) const
  {
    const unsigned int n_points = points.size();

    Assert (values.size() == n_points,
            ExcDimensionMismatch (values.size(), n_points));

    Assert (component == 0,
            ExcIndexRange (component, 0, 1));

    for (unsigned int i=0; i<n_points; ++i)
    {
        values[i] = PerfectlyMatchedLayer::lambda22(points[i]);
    }
  }

  template<int dim>
  std::complex<double> PerfectlyMatchedLayer<dim>::lambda33 (const Point<dim> &p,
                                     const unsigned int /*component*/) const
  {
	     std::complex<double> s1(1, sigma(p[0]));
	     std::complex<double> s2(1, sigma(p[1]));
		  switch (dim)
		  {
		  case 3: {std::complex<double> s3(1, sigma(p[2])); return (s1 * s2) / s3;}
		  default:
			  abort ();
		  }
  }

  template <int dim>
  void PerfectlyMatchedLayer<dim>::lambda33_list (const std::vector<Point<dim>> &points,
                                     std::vector<std::complex<double>>            &values,
                                     const unsigned int              component) const
  {
    const unsigned int n_points = points.size();

    Assert (values.size() == n_points,
            ExcDimensionMismatch (values.size(), n_points));

    Assert (component == 0,
            ExcIndexRange (component, 0, 1));

    for (unsigned int i=0; i<n_points; ++i)
    {
        values[i] = PerfectlyMatchedLayer::lambda33(points[i]);
    }

  }

   template < int dim>
   double PerfectlyMatchedLayer<dim>::sigma(double x) const
   {
  	 double c = 1.0,  sigma_star = 1 * c;
  	 double omega = 1e1;
  	 double a_star = 1;
  	 double dx = 2 / pow(2,7);
  	 int N = 8;
  	 double a = 1 - dx * N;
  	 if (x >= a /*|| x <= -a*/)
  	 {
  	     return pow((abs(x) - a)/ (a_star - a), 2) * sigma_star/ omega;
  	 }
  	 else
  	 {
  		 return 0;
  	 }
   }


   // This class defines the weight function. It is used for computing the L2 error
   // In our project we restrict the integration domain to non-PML region.

   template <int dim>
   class InteriorWeightFunction : public Function<dim>
   {
   public:
 	  InteriorWeightFunction ():Function<dim>() {}
 	  double value (const Point<dim> &p,
 			  const unsigned int /*component*/) const
 	  {
 		  if (p.norm()<=0.5)
 			  return 1;
 		  else
 			  return 0;
 	  }
   };

   // The actual definition of the values and gradients of the exact solution
   // class is according to their mathematical definition and does not need
   // much explanation.
   //

   template <int dim>
   class Solution : public Function<dim>
   {
   public:
     Solution () : Function<dim>(2) {}
     double value (const Point<dim>   &p,
     		const unsigned int  component) const;

   };


   template <int dim>
   double Solution<dim>::value (const Point<dim>   &p,
                                const unsigned int component) const
   {
 	  if (dim == 2)
 	  {
 		  switch (component)
 		  {
 		  case 0: return cos(wave_number_2d * p);
 		  case 1: return -sin(wave_number_2d * p);
 		  default:
 			  abort ();
 		  }
 	  }
   }


   // @sect3{The <code>ScatteringProblem</code> class}

   // Finally here is the main class of this program.  It's very similar to  step-29.
   // The ParameterHandler object that is passed to the constructor is stored
   // as a reference to allow easy access to the parameters from all functions
   // of the class.  Since we are working with vector valued finite elements,
   // the FE object we are using is of type FESystem.
  template <int dim>
  class ScatteringProblem
  {
  public:
    ScatteringProblem (ParameterHandler &);
    ~ScatteringProblem ();
    void run ();

  private:
    void make_grid ();
    void setup_system ();
    void assemble_system ();
    void solve ();
    void output_results () const;
    void process_solution (/*const unsigned int cycle*/);

    ParameterHandler      &prm;

    Triangulation<dim>     triangulation;
    DoFHandler<dim>        dof_handler;
    FESystem<dim>          fe;
    SparsityPattern        sparsity_pattern;
    SparseMatrix<double>   system_matrix;
    Vector<double>         solution, system_rhs;
  };

  template <int dim>
  void ScatteringProblem<dim>::process_solution (/*const unsigned int cycle*/)
  {
    Vector<float> difference_per_cell (triangulation.n_active_cells());
    InteriorWeightFunction<dim> w1;
    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       Solution<dim>(),
                                       difference_per_cell,
                                       QGauss<dim>(3),
                                       VectorTools::L2_norm,
                                       &w1);
    const double L2_error = difference_per_cell.l2_norm();
//    VectorTools::integrate_difference (dof_handler,
//                                       solution,
//                                       Solution<dim>(),
//                                       difference_per_cell,
//                                       QGauss<dim>(3),
//                                       VectorTools::H1_seminorm);
//    const double H1_error = difference_per_cell.l2_norm();
    const QTrapez<1>     q_trapez;
    const QIterated<dim> q_iterated (q_trapez, 5);
    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       Solution<dim>(),
                                       difference_per_cell,
                                       q_iterated,
                                       VectorTools::Linfty_norm,
                                       &w1);
    const double Linfty_error = difference_per_cell.linfty_norm();
    const unsigned int n_active_cells=triangulation.n_active_cells();
    const unsigned int n_dofs=dof_handler.n_dofs();
    std::cout /*<< "Cycle " << cycle << ':'
              << std::endl*/
              << "   Number of active cells:       "
              << n_active_cells
              << std::endl
              << "   Number of degrees of freedom: "
              << n_dofs
              << std::endl
              <<" L2 Error: "
              << L2_error
              << std::endl
              <<" Linfty Error: "
              << Linfty_error
              << std::endl;
  }

  // The constructor takes the ParameterHandler object and stores it in a
  // reference. It also initializes the DoF-Handler and the finite element
  // system, which consists of two copies of the scalar Q1 field, one for $v$
  // and one for $w$:
  template <int dim>
  ScatteringProblem<dim>::ScatteringProblem (ParameterHandler  &param)
    :
    prm(param),
    dof_handler(triangulation),
    fe(FE_Q<dim>(1), 2)
  {}


  template <int dim>
  ScatteringProblem<dim>::~ScatteringProblem ()
  {
    dof_handler.clear();
  }


  // @sect4{<code>ScatteringProblem::make_grid</code>}

  // Here we setup the grid for our domain.  As mentioned in the exposition,
  // the geometry is just a unit square (in 2d)
  template <int dim>
  void ScatteringProblem<dim>::make_grid ()
  {
    deallog << "Generating grid... ";
    Timer timer;
    timer.start ();

    GridGenerator::subdivided_hyper_cube (triangulation, 5, -1, 1);
    typename Triangulation<dim>::cell_iterator
    cell = triangulation.begin (),
    endc = triangulation.end();

    for (; cell!=endc; ++cell)
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
      {
    	  if ( cell->face(face)->at_boundary() && ((cell->face(face)->center()[0] + 1)*(cell->face(face)->center()[0] + 1) < 0.01))
          {
    		  cell->face(face)->set_boundary_indicator (1);
          }
      }

    triangulation.set_boundary(1);
    //GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global (4);

    timer.stop ();
    deallog << "done ("
            << timer()
            << "s)"
            << std::endl;

    deallog << "  Number of active cells:  "
            << triangulation.n_active_cells()
            << std::endl;
  }


  // @sect4{<code>ScatteringProblem::setup_system</code>}
  //
  // Initialization of the system matrix, sparsity patterns and vectors are
  // the same as in previous examples and therefore do not need further
  // comment. As in the previous function, we also output the run time of what
  // we do here:
  template <int dim>
  void ScatteringProblem<dim>::setup_system ()
  {
    deallog << "Setting up system... ";
    Timer timer;
    timer.start();

    dof_handler.distribute_dofs (fe);

    sparsity_pattern.reinit (dof_handler.n_dofs(),
                             dof_handler.n_dofs(),
                             dof_handler.max_couplings_between_dofs());

    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
    sparsity_pattern.compress();

    system_matrix.reinit (sparsity_pattern);
    system_rhs.reinit (dof_handler.n_dofs());
    solution.reinit (dof_handler.n_dofs());

    timer.stop ();
    deallog << "done ("
            << timer()
            << "s)"
            << std::endl;

    deallog << "  Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;
  }



  // @sect4{<code>ScatteringProblem::assemble_system</code>}

  // As before, this function takes care of assembling the system matrix and
  // right hand side vector:
  template <int dim>
  void ScatteringProblem<dim>::assemble_system ()
  {
    deallog << "Assembling system matrix... ";
    Timer timer;
    timer.start ();

    QGauss<dim>    quadrature_formula(2);
    QGauss<dim-1>  face_quadrature_formula(2);

    const unsigned int n_q_points       = quadrature_formula.size(),
                      /*n_face_q_points  = face_quadrature_formula.size(),*/
                       dofs_per_cell    = fe.dofs_per_cell;


    const RefractiveIndex<dim> refractive_index;
    std::vector<double>    refractive_index_values (n_q_points);

    const RightHandSide<dim> right_hand_side;
    std::vector<std::complex<double>>  rhs_values (n_q_points);

    FEValues<dim>  fe_values (fe, quadrature_formula,
                              update_values | update_gradients | update_q_points |
                              update_JxW_values);

    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                      update_values | update_JxW_values);

    const PerfectlyMatchedLayer<dim> PML;
    std::vector<std::complex<double>>    pml_lambda00_values (n_q_points);
    std::vector<std::complex<double>>    pml_lambda11_values (n_q_points);
    std::vector<std::complex<double>>    pml_lambda22_values (n_q_points);
    std::vector<std::complex<double>>    pml_lambda33_values (n_q_points);

    FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

    for (; cell!=endc; ++cell)
      {

    	fe_values.reinit (cell);
        cell_matrix = 0;
        cell_rhs = 0;

        refractive_index.value_list (fe_values.get_quadrature_points(),
       			refractive_index_values);
        right_hand_side.value_list (fe_values.get_quadrature_points(),
                                           rhs_values);

        //common coefficients in PML for both 2d and 3d
        PML.lambda00_list (fe_values.get_quadrature_points(),
       			pml_lambda00_values);
        PML.lambda11_list (fe_values.get_quadrature_points(),
       			pml_lambda11_values);
        PML.lambda22_list (fe_values.get_quadrature_points(),
       			pml_lambda22_values);
        if(dim == 3)
        {
            PML.lambda33_list (fe_values.get_quadrature_points(),pml_lambda33_values);
        }

        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
        	const unsigned int

        	component_i = fe.system_to_component_index(i).first;

           for (unsigned int j=0; j<dofs_per_cell; ++j)
           {
           	    const unsigned int
           	    component_j = fe.system_to_component_index(j).first;
           	    //this is for the real part of the helmholtz equation.
            	if (component_i ==  component_j)
            	{
                    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                    {
                		Tensor<2,dim> coefficients;
                		double coefficient0;
                		//It seems more elegant to define pml coefficient in tensor form.
                		//However, as stated before, it is more convenient(if we extend our scalar field to a Maxwell case
                		// to define the parameter in
                		// each dimension separately.
                		switch (dim)
                		{
                		case 2:
                			coefficient0 = pml_lambda00_values[q_point].real();
                    		coefficients[0][0] = pml_lambda11_values[q_point].real();
                    		coefficients[1][1] = pml_lambda22_values[q_point].real();
                    		break;
                		case 3:
                			coefficient0 = pml_lambda00_values[q_point].real();
                    		coefficients[0][0] = pml_lambda11_values[q_point].real();
                    		coefficients[1][1] = pml_lambda22_values[q_point].real();
                    		coefficients[2][2] = pml_lambda33_values[q_point].real();
                    		break;
                		default:
                			Assert (false, ExcNotImplemented());
                		}
                	// we have used a scattered field formulation in our system.
                      cell_matrix(i,j) += (((fe_values.shape_value(i,q_point) * fe_values.shape_value(j,q_point)) *
                                            (- wave_number_2d * wave_number_2d * coefficient0
                                             /* refractive_index_values[q_point]*/)
                                        +   (fe_values.shape_grad(i,q_point) * coefficients * fe_values.shape_grad(j,q_point)) ) *
                                             fe_values.JxW(q_point));
                    }
            	}
            	// this is the imaginary part of the helmholtz equation.
            	else
            	{
                    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                    {
                		Tensor<2,dim> coefficients;
                		double coefficient0;
                		switch (dim)
                		{
                		case 2:
                			coefficient0 = pml_lambda00_values[q_point].imag();
                    		coefficients[0][0] = pml_lambda11_values[q_point].imag();
                    		coefficients[1][1] = pml_lambda22_values[q_point].imag();
                    		break;
                		case 3:
                			coefficient0 = pml_lambda00_values[q_point].imag();
                    		coefficients[0][0] = pml_lambda11_values[q_point].imag();
                    		coefficients[1][1] = pml_lambda22_values[q_point].imag();
                    		coefficients[2][2] = pml_lambda33_values[q_point].imag();
                    		break;
                		default:
                			Assert (false, ExcNotImplemented());
                		}
                        cell_matrix(i,j) += ((component_j == 0) ? -1 : 1) *
                        		            (((fe_values.shape_value(i,q_point) * fe_values.shape_value(j,q_point)) *
                                              (- wave_number_2d * wave_number_2d * coefficient0
                                              /* refractive_index_values[q_point]*/)
                                          + (fe_values.shape_grad(i,q_point) * coefficients *fe_values.shape_grad(j,q_point))) *
                                              fe_values.JxW(q_point));
                    }
            	}
           }
        }

        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            system_matrix.add (local_dof_indices[i],
                               local_dof_indices[j],
                               cell_matrix(i,j));

        }
    }


    std::map<types::global_dof_index,double> boundary_values;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              1,
                                              DirichletBoundaryValues<dim>(),
                                              boundary_values);


    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        solution,
                                        system_rhs);

    timer.stop ();
    deallog << "done ("
    		<< timer()
            << "s)"
            << std::endl;
  }


  // @sect4{<code>ScatteringProblem::solve</code>}

  // As already mentioned in the introduction, the system matrix is neither
  // symmetric nor definite, and so it is not quite obvious how to come up
  // with an iterative solver and a preconditioner that do a good job on this
  // matrix.  We chose instead to go a different way and solve the linear
  // system with the sparse LU decomposition provided by UMFPACK. This is
  // often a good first choice for 2D problems and works reasonably well even
  // for a large number of DoFs.  The deal.II interface to UMFPACK is given by
  // the SparseDirectUMFPACK class, which is very easy to use and allows us to
  // solve our linear system with just 3 lines of code.

  // Note again that for compiling this example program, you need to have the
  // deal.II library built with UMFPACK support, which can be achieved by
  // providing the <code> --with-umfpack</code> switch to the configure script
  // prior to compilation of the library.
  template <int dim>
  void ScatteringProblem<dim>::solve ()
  {
    deallog << "Solving linear system... ";
    Timer timer;
    timer.start ();

    SparseDirectUMFPACK  A_direct;
    A_direct.initialize(system_matrix);

    A_direct.vmult (solution, system_rhs);

    timer.stop ();
    deallog << "done ("
            << timer ()
            << "s)"
            << std::endl;
  }



  // @sect4{<code>ScatteringProblem::output_results</code>}

   // Here we output our solution $v$ and $w$ as well as the derived quantity
   // $|u|$ in the format specified in the parameter file. Most of the work for
   // deriving $|u|$ from $v$ and $w$ was already done in the implementation of
   // the <code>ComputeIntensity</code> class, so that the output routine is
   // rather straightforward and very similar to what is done in the previous
   // tutorials.
  template <int dim>
  void ScatteringProblem<dim>::output_results () const
  {
    deallog << "Generating output... ";
    Timer timer;
    timer.start ();

    ComputeIntensity<dim> intensities;
    /*DataOut<dim> data_out;

    data_out.attach_dof_handler (dof_handler);

    prm.enter_subsection("Output parameters");

    const std::string output_file    = prm.get("Output file");
    data_out.parse_parameters(prm);

    prm.leave_subsection ();

    const std::string filename = output_file +
                                 data_out.default_suffix();

    std::ofstream output (filename.c_str());

    std::vector<std::string> solution_names;
    solution_names.push_back ("Re_u");
    solution_names.push_back ("Im_u");

    data_out.add_data_vector (solution, solution_names);

    data_out.add_data_vector (solution, intensities);

    data_out.build_patches ();
    data_out.write (output);*/

    //DataOutBase::EpsFlags eps_flags;
    //eps_flags.z_scaling = 1;

    DataOut<dim> data_out;
    //data_out.set_flags (eps_flags);

    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, intensities);
    data_out.build_patches ();

    std::ofstream output ("intensities.gpl");
    data_out.write_gnuplot (output);


    timer.stop ();
    deallog << "done ("
            << timer()
            << "s)"
            << std::endl;
  }



  // @sect4{<code>ScatteringProblem::run</code>}

  // Here we simply execute our functions one after the other:
  template <int dim>
  void ScatteringProblem<dim>::run ()
  {
    make_grid ();
    setup_system ();
    assemble_system ();
    solve ();
    output_results ();
    process_solution ();
  }
}


// @sect4{The <code>main</code> function}

// Finally the <code>main</code> function of the program.
int main ()
{
  try
    {
      using namespace dealii;
      using namespace Step29;

      ParameterHandler  prm;
      ParameterReader   param(prm);
      param.read_parameters("step-29.prm");

      wave_number_2d[0] = 1e1;
      wave_number_2d[1] = 0e1;
      ScatteringProblem<2>  planewave_2d (prm);
      planewave_2d.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
