#pragma once

#include <unistd.h>

static int
read_all(int fd, void *output, size_t size)
{
  auto begin = static_cast<char *>(output);
  auto end = begin + size;

  while (begin != end)
  {
    auto ret = read(fd, begin, end - begin);

    if (ret <= 0)
      return ret;

    begin += ret;
  }

  return 0;
}
