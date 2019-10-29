#pragma once

#include <chrono>
#include <fstream>
#include <map>
#include <sstream>
#include <thread>
#include <vector>

#include <jitify.hpp>
#define NO_FREETYPE
#include <pngwriter.h>
#include <xeus/xjson.hpp>
#include <xtl/xbase64.hpp>

#include "color_maps.hpp"

// checks whether the return value was successful
// If not, print an error message
inline void cuCheck(cudaError_t code) {
  if (code != cudaSuccess) {
    std::cerr << "Error code: " << code << std::endl
              << cudaGetErrorString(code) << std::endl;
  }
}

inline void cuCheck(cudaError_enum code) {
  if (code != CUDA_SUCCESS) {
    const char *str;
    cuGetErrorName(code, &str);
    std::cerr << "Error code: " << code << std::endl << str << std::endl;
  }
}

static jitify::JitCache kernel_cache;

class Base_task {
  friend class Task1;
  friend class Task2;
  friend class Task3;

private:
  // dimension of the world
  int dim;
  // dim + 2 (ghostcells)
  int world_size;
  // initial world on CPU
  int *init_world;
  // memory on the CPU, to copy back a result
  int *host_world;
  // memory on GPU
  // origin world of iteration
  int *d_world;
  // new world of iteration
  int *d_new_world;
  // swap pointer
  int *d_swap;

  // scale factor for the pictures
  // if 1, a cell has the size of a pixel
  unsigned int scale = 20;
  // get a cell state and returns a respective color
  Color_map<int> *color_map;
  // store all result pictures
  std::vector<std::vector<unsigned char>> pngs;
  // activates a delay between all pictures that print_worlds() displays
  bool sleep;
  // delay time in ms for print_worlds()
  unsigned sleep_time = 800;

  std::map<std::string, std::unique_ptr<jitify::KernelInstantiation>> instances;

  ///\brief generate a png of the world in the variable host_world
  ///
  ///\param [in] ghostcells - if true, print also the ghostcells
  ///\param [in] scale - If 1, a cell has the size of a pixel
  ///
  void generate_png(bool ghostcells, unsigned int scale) {
    bool g = !ghostcells;
    std::vector<unsigned char> png;
    pngwriter pic((world_size - g * 2) * scale, (world_size - g * 2) * scale, 0,
                  "tmp.png");

    // true == (int)1 -> if ghostcells true, indicies starts with 1 ends with
    // height-1
    for (unsigned int y = g; y < world_size - g; ++y) {
      for (unsigned int x = g; x < world_size - g; ++x) {
        for (unsigned scale_y = 0; scale_y < scale; ++scale_y) {
          for (unsigned scale_x = 0; scale_x < scale; ++scale_x) {
            pic.plot(((x - g) * scale) + scale_x,
                     pic.getwidth() - (((y - g) * scale) + scale_y),
                     color_map->r(host_world[y * world_size + x]),
                     color_map->g(host_world[y * world_size + x]),
                     color_map->b(host_world[y * world_size + x]));
          }
        }
      }
    }

    pic.write_to_buffer(png);
    pngs.push_back(png);
  }

public:
  ///\brief Constructor
  ///
  ///\param [in] dim - size of the world
  ///\param [in] sleep - do break between the pictures of print_worlds()
  ///
  Base_task(unsigned int dim, bool sleep)
      : dim(dim), world_size(dim + 2), sleep(sleep),
        color_map(new Black_white_map<int>) {
    host_world = new int[world_size * world_size];
    init_world = new int[world_size * world_size];
    cuCheck(
        cudaMalloc((void **)&d_world, sizeof(int) * world_size * world_size));
    cuCheck(cudaMalloc((void **)&d_new_world,
                       sizeof(int) * world_size * world_size));
  }

  ~Base_task() {
    delete color_map;
    delete[] host_world;
    delete[] init_world;
    cudaFree(d_world);
    cudaFree(d_new_world);
  }

  void set_scale(unsigned int s) { scale = s; }
  void set_sleep_time(unsigned int st) { sleep_time = st; }

  ///\brief Change the dimension of the world. The function deletes all worlds
  /// on the host and device.
  ///
  ///\param [in] dim - size of the world
  ///
  void set_dim(unsigned int dim) {
    this->dim = dim;
    this->world_size = dim + 2;

    delete[] host_world;
    delete[] init_world;
    cudaFree(d_world);
    cudaFree(d_new_world);

    host_world = new int[world_size * world_size];
    init_world = new int[world_size * world_size];
    cuCheck(
        cudaMalloc((void **)&d_world, sizeof(int) * world_size * world_size));
    cuCheck(cudaMalloc((void **)&d_new_world,
                       sizeof(int) * world_size * world_size));
  }

  ///\brief Loads a first world from a file. The world from the file is placed
  /// in the upper left corner. If a line in the file is longer than dim, it is
  /// truncated. The same happens if the number of lines is greater than dim. If
  /// a row is smaller than dim or the number of rows is smaller than dim, the
  /// uncovered cells keep their value (initialize init_word with zeros before
  /// to avoid unexpected behavior).
  ///
  ///\param [in] filename - path of the input file
  ///
  void load_world(std::string filename) {
    std::ifstream input(filename.c_str());
    if (!input.is_open()) {
      std::cerr << "can't open file: " << filename << std::endl;
      return;
    }

    int row = 1, column = 1;
    std::string line;
    while (std::getline(input, line)) {
      std::stringstream ss(line);
      std::string element;
      while (ss >> element) {
        if (column == (dim + 1)) {
          column = 1;
          break;
        }
        init_world[row * (dim + 2) + column] =
            (element == "x" || element == "X" || element == "1") ? 1 : 0;
        ++column;
      }
      column = 1;
      ++row;
      if (row == (dim + 1))
        break;
    }
    input.close();
  }

  ///\brief set all values in init_world to 0
  ///
  void init_zero() {
    for (int y = 0; y < world_size; ++y) {
      for (int x = 0; x < world_size; ++x) {
        init_world[y * (world_size) + x] = 0;
      }
    }
  }

  ///\brief print all values of init_world
  ///
  void print_init() {
    for (int y = 0; y < world_size; ++y) {
      for (int x = 0; x < world_size; ++x) {
        std::cout << init_world[y * (world_size) + x] << " ";
      }
      std::cout << std::endl;
    }
  }

  ///\brief clear pngs and copy init world to GPU
  ///
  void init() {
    pngs.clear();
    cuCheck(cudaMemcpy(d_world, init_world,
                       sizeof(int) * world_size * world_size,
                       cudaMemcpyHostToDevice));
  }

  ///\brief compile the GPU kernel
  ///
  ///\param [in] kernel_path - path to the kernel
  ///\param [in] kernel_name - name of the kernel
  ///\param [in] targs - Template parameter of the kernel
  ///
  ///\returns 0 = success, -1 can't open file
  ///
  template <typename... TemplateArgs>
  int compile_kernel(const std::string kernel_path,
                     const std::string kernel_name, TemplateArgs... targs) {
    std::ifstream kernel_file(kernel_path.c_str());
    if (!kernel_file.is_open()) {
      std::cerr << "can't open file: " << kernel_path << std::endl;
      return -1;
    }

    std::string kernel_code;
    getline(kernel_file, kernel_code, (char)kernel_file.eof());
    kernel_file.close();

    jitify::Program prog = kernel_cache.program(kernel_code);

    if (instances.count(kernel_name) > 0) {
      instances[kernel_name] = std::make_unique<jitify::KernelInstantiation>(
          prog.kernel(kernel_name).instantiate(targs...));
    } else {
      instances.insert(std::make_pair(
          kernel_name, std::make_unique<jitify::KernelInstantiation>(
                           prog.kernel(kernel_name).instantiate(targs...))));
    }

    return 0;
  }

  ///\brief run the compiled kernel
  ///
  ///\param [in] blocks - cuda block size
  ///\param [in] threads - cuda thread size
  ///
  inline void run_kernel(dim3 blocks, dim3 threads) {
    cuCheck(cudaMemcpy(host_world, d_world,
                       sizeof(int) * world_size * world_size,
                       cudaMemcpyDeviceToHost));
    generate_png(false, scale);
  }

  ///\brief print all pictures in pngs at the notebook
  ///
  ///\param [in] clear_output - if true, replace the picture with the next one
  ///            in the notebook, otherwise print every picture under his
  ///            predecessor
  ///
  void print_worlds(bool clear_output) {
    // memory objects for output in the web browser
    std::stringstream buffer;
    xeus::xjson mine;

    for (auto png : pngs) {
      if (clear_output)
        xeus::get_interpreter().clear_output(true);

      buffer.str("");
      for (auto c : png) {
        buffer << c;
      }

      mine["image/png"] = xtl::base64encode(buffer.str());
      xeus::get_interpreter().display_data(
          std::move(mine), xeus::xjson::object(), xeus::xjson::object());

      if (sleep)
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
    }
  }
};

class Task1 : public Base_task {
public:
  Task1(unsigned int dim) : Base_task(dim, false) {}

  ///\brief run the compiled ghostCols kernel
  ///
  ///\param [in] kernel_name - name of the kernel
  ///\param [in] blocks - cuda block size
  ///\param [in] threads - cuda thread size
  ///
  inline void run_kernel(std::string kernel_name, dim3 blocks, dim3 threads) {
    if (instances.count(kernel_name) == 0) {
      std::cerr << kernel_name << " was not compiled" << std::endl;
      return;
    }
    cuCheck(instances[kernel_name]
                ->configure(blocks, threads)
                .launch(dim, d_world));
    cuCheck(cudaDeviceSynchronize());
    cuCheck(cudaGetLastError());

    cuCheck(cudaMemcpy(host_world, d_world,
                       sizeof(int) * world_size * world_size,
                       cudaMemcpyDeviceToHost));
    generate_png(true, scale);
  }

  ///\brief generate a picture, which contains solution, the calculated world
  ///       and a picture, which contains the difference between the solution
  ///       and the result
  ///
  void gen_diff() {
    // ghostcells
    std::vector<unsigned char> png;
    // width: 3 pictures + 2 * 4 pixel space
    pngwriter pic((world_size + 4 + world_size + 4 + world_size) * scale,
                  world_size * scale, 0, "tmp.png");

    std::cout << pic.getheight() << std::endl << pic.getwidth() << std::endl;

    // set all cells to gray
    for (unsigned int y = 0; y < pic.getheight(); ++y) {
      for (unsigned int x = 0; x < pic.getwidth(); ++x) {
        pic.plot(x, y, 55000, 55000, 55000);
      }
    }

    int solution[] = {1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1,
                      0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                      1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
                      0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1,
                      0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1};

    // generate picture with solution on the left side of the picture
    for (unsigned int y = 0; y < world_size; ++y) {
      for (unsigned int x = 0; x < world_size; ++x) {
        for (unsigned scale_y = 0; scale_y < scale; ++scale_y) {
          for (unsigned scale_x = 0; scale_x < scale; ++scale_x) {
            pic.plot((x * scale) + scale_x,
                     (world_size * scale) - ((y * scale) + scale_y),
                     color_map->r(solution[y * world_size + x]),
                     color_map->g(solution[y * world_size + x]),
                     color_map->b(solution[y * world_size + x]));
          }
        }
      }
    }

    // generate picture with calculated result in the middle of the picture
    for (unsigned int y = 0; y < world_size; ++y) {
      for (unsigned int x = 0; x < world_size; ++x) {
        for (unsigned scale_y = 0; scale_y < scale; ++scale_y) {
          for (unsigned scale_x = 0; scale_x < scale; ++scale_x) {
            pic.plot((world_size + 4) * scale + (x * scale) + scale_x,
                     (world_size * scale) - ((y * scale) + scale_y),
                     color_map->r(host_world[y * world_size + x]),
                     color_map->g(host_world[y * world_size + x]),
                     color_map->b(host_world[y * world_size + x]));
          }
        }
      }
    }

    Ghost_diff_map diff_map;

    // generate picture with differenced on the right side of the picture
    for (unsigned int y = 0; y < world_size; ++y) {
      for (unsigned int x = 0; x < world_size; ++x) {
        for (unsigned scale_y = 0; scale_y < scale; ++scale_y) {
          for (unsigned scale_x = 0; scale_x < scale; ++scale_x) {
            pic.plot(2 * (world_size + 4) * scale + (x * scale) + scale_x,
                     (world_size * scale) - ((y * scale) + scale_y),
                     diff_map.r(solution[y * world_size + x] * 10 +
                                host_world[y * world_size + x]),
                     diff_map.g(solution[y * world_size + x] * 10 +
                                host_world[y * world_size + x]),
                     diff_map.b(solution[y * world_size + x] * 10 +
                                host_world[y * world_size + x]));
          }
        }
      }
    }

    pic.write_to_buffer(png);
    pngs.push_back(png);
  }
};

class Task2 : public Base_task {
public:
  Task2(unsigned int dim) : Base_task(dim, true) {}

  void init() {
    Base_task::init();
    for (int i = 0; i < world_size * world_size; ++i)
      host_world[i] = init_world[i];
    generate_png(true, scale);
  }

  ///\brief run the ghost and GOL kernel
  ///
  ///\param [in] ghost_row_kernel_name - name of the Ghostcopy Row kernel
  ///\param [in] row_blocks - cuda block size of the row ghost kernel
  ///\param [in] row_threads - cuda thread size of the row ghost kernel
  ///\param [in] ghost_col_kernel_name - name of the Ghostcopy Column kernel
  ///\param [in] col_blocks - cuda block size of the column ghost kernel
  ///\param [in] col_threads - cuda thread size of the column ghost kernel
  ///\param [in] GOL_kernel_name - name of the Game of Life kernel
  ///\param [in] gol_blocks - cuda block size of the GOL kernel
  ///\param [in] gol_threads - cuda thread size of the GOL kernel
  ///
  inline void run_kernel(std::string ghost_row_kernel_name, dim3 row_blocks,
                         dim3 row_threads, std::string ghost_col_kernel_name,
                         dim3 col_blocks, dim3 col_threads,
                         std::string GOL_kernel_name, dim3 gol_blocks,
                         dim3 gol_threads) {
    if (instances.count(ghost_row_kernel_name) == 0) {
      std::cerr << ghost_row_kernel_name << " was not compiled" << std::endl;
      return;
    }

    if (instances.count(ghost_col_kernel_name) == 0) {
      std::cerr << ghost_col_kernel_name << " was not compiled" << std::endl;
      return;
    }

    if (instances.count(GOL_kernel_name) == 0) {
      std::cerr << GOL_kernel_name << " was not compiled" << std::endl;
      return;
    }

    cuCheck(instances[ghost_row_kernel_name]
                ->configure(row_blocks, row_threads)
                .launch(dim, d_world));
    cuCheck(instances[ghost_col_kernel_name]
                ->configure(col_blocks, col_threads)
                .launch(dim, d_world));
    cuCheck(instances[GOL_kernel_name]
                ->configure(gol_blocks, gol_threads)
                .launch(dim, d_world, d_new_world));
    cuCheck(cudaDeviceSynchronize());
    cuCheck(cudaGetLastError());

    d_swap = d_new_world;
    d_new_world = d_world;
    d_world = d_swap;

    cuCheck(cudaMemcpy(host_world, d_world,
                       sizeof(int) * world_size * world_size,
                       cudaMemcpyDeviceToHost));
    generate_png(true, scale);
  }

  ///\brief run n iterations the kernel and measure the time
  ///
  ///\param [in] iteration - number of iterations
  ///\param [in] ghost_row_kernel_name - name of the Ghostcopy Row kernel
  ///\param [in] row_blocks - cuda block size of the row ghost kernel
  ///\param [in] row_threads - cuda thread size of the row ghost kernel
  ///\param [in] ghost_col_kernel_name - name of the Ghostcopy Column kernel
  ///\param [in] col_blocks - cuda block size of the column ghost kernel
  ///\param [in] col_threads - cuda thread size of the column ghost kernel
  ///\param [in] GOL_kernel_name - name of the Game of Life kernel
  ///\param [in] gol_blocks - cuda block size of the GOL kernel
  ///\param [in] gol_threads - cuda thread size of the GOL kernel
  ///
  double bench_kernel(unsigned int iteration, std::string ghost_row_kernel_name,
                      dim3 row_blocks, dim3 row_threads,
                      std::string ghost_col_kernel_name, dim3 col_blocks,
                      dim3 col_threads, std::string GOL_kernel_name,
                      dim3 gol_blocks, dim3 gol_threads) {
    if (instances.count(ghost_row_kernel_name) == 0) {
      std::cerr << "ghostRows was not compiled" << std::endl;
      return -1.0;
    }

    if (instances.count(ghost_col_kernel_name) == 0) {
      std::cerr << "ghostCols was not compiled" << std::endl;
      return -1.0;
    }

    if (instances.count(GOL_kernel_name) == 0) {
      std::cerr << GOL_kernel_name << " was not compiled" << std::endl;
      return -1.0;
    }

    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> time;

    start = std::chrono::system_clock::now();
    for (unsigned int i = 0; i < iteration; ++i) {
      cuCheck(instances[ghost_row_kernel_name]
                  ->configure(row_blocks, row_threads)
                  .launch(dim, d_world));
      cuCheck(instances[ghost_col_kernel_name]
                  ->configure(col_blocks, col_threads)
                  .launch(dim, d_world));
      cuCheck(instances[GOL_kernel_name]
                  ->configure(gol_blocks, gol_threads)
                  .launch(dim, d_world, d_new_world));

      d_swap = d_new_world;
      d_new_world = d_world;
      d_world = d_swap;
    }
    cuCheck(cudaDeviceSynchronize());
    end = std::chrono::system_clock::now();

    time = end - start;

    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
  }
};

class Task3 : public Task2 {
public:
  Task3(unsigned int dim) : Task2(dim) {}
};
